#!/usr/bin/env python3
import os                                  # Manage environment variables[1]
import sys                                 # Modify Python import path[1]

# Suppress TensorFlow and plugin warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Hide TF INFO/WARN/ERROR logs[2]
os.environ["TRANSFORMERS_NO_TF"]   = "1"   # Use PyTorch-only in Transformers[3]

# Add local Ultravox code path (adjust if you cloned elsewhere)
sys.path.insert(0, os.getcwd())
import io                                                      # In-memory byte streams[4]
import yaml                                                    # YAML configuration[5]
import torch                                                   # Device management[6]
import soundfile as sf                                         # Audio I/O[7]
import numpy as np                                             # Numeric arrays

from fastapi import FastAPI, UploadFile, File                  # API framework[8]
from scripts.preprocess_audio import preprocess_audio           # Audio resampling & normalization[9]
from utils.audio_vad import split_audio                        # WebRTC VAD segmentation[10]
from utils.emotion_classifier import EmotionClassifier          # Emotion classification pipeline[11]

# Now import UltravoxInference from the local codebase
from ultravox.inference.ultravox_infer import UltravoxInference # STT + LLM inference[12]

# Kokoro TTS pipeline for expressive speech synthesis
from kokoro import KPipeline                                    # GPU-accelerated TTS[13]

# Load YAML configuration file
config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")
with open(config_path, "r") as cfg:
    config = yaml.safe_load(cfg)    # Contains model paths, device, VAD settings[5]

# Select inference device ('cuda' or 'cpu')
device = torch.device(config.get("device", "cpu"))  # e.g., 'cuda:0'[6]

# Initialize UltravoxInference (positional args only)
stt_llm = UltravoxInference(
    config["model"]["stt_llm_path"],  # Path to Ultravox model directory
    device=device                     # GPU or CPU device
)

# Initialize Kokoro TTS and EmotionClassifier
tts_model     = KPipeline(lang_code='a')                        # 'a' = American English[13]
emotion_model = EmotionClassifier(
    config["emotion_classifier"]["model_name"],
    device=config.get("device", "cpu")
)

# Create FastAPI app
app = FastAPI(title="Real-Time Emotional STS Agent")

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    """
    1. Read uploaded audio into numpy array  
    2. Segment speech via VAD  
    3. Preprocess each segment (resample & normalize)  
    4. Perform STT + emotion classification + LLM response  
    5. Synthesize reply with Kokoro TTS  
    6. Return text, emotion, and hex-encoded WAV for each segment  
    """
    # Read raw audio
    data, sr = sf.read(io.BytesIO(await file.read()))  # WAV/FLAC → numpy array[7]

    # Split into voiced segments
    segments = split_audio(data, sr, config["vad"]["aggressiveness"])  # VAD[10]

    results = []
    for seg in segments:
        # Resample & normalize
        seg_norm = preprocess_audio(seg, sr)  # 16 kHz, amplitude [-1,1][9]

        # STT → transcript
        transcript = stt_llm({"audio": seg_norm, "sampling_rate": sr})[0]["text"]  # HF-style API[12]

        # Emotion detection
        emotion = emotion_model.predict(seg_norm, sr)  # e.g., 'happiness'[11]

        # Generate LLM reply conditioned on emotion
        reply = stt_llm.generate_response(prompt=transcript, emotion=emotion)  # Custom method[12]

        # Synthesize speech
        tts_output = tts_model(reply, voice='af_heart')  # Returns (gen, pros, audio)[13]
        audio_arr = tts_output[0][2]

        # Encode WAV bytes to hex string
        buf = io.BytesIO()
        sf.write(buf, audio_arr, config.get("tts_sample_rate", 24000), format="WAV")
        hex_audio = buf.getvalue().hex()

        results.append({
            "user_text":  transcript,
            "emotion":    emotion,
            "agent_text": reply,
            "audio_hex":  hex_audio
        })

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),  # Default to 8000 if PORT not set
        reload=True                         # Auto-reload on code changes[8]
    )
