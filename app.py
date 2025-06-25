#!/usr/bin/env python3
import os                                                      # Manage environment variables[1]
# Suppress TensorFlow logs and disable TF backend in Transformers
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"                       # Hide TensorFlow INFO/WARNING/ERROR[2]
os.environ["TRANSFORMERS_NO_TF"]   = "1"                       # Force PyTorch-only pipelines[3]

import io                                                      # Byte streams for in-memory I/O[4]
import yaml                                                    # YAML configuration parsing[5]
import torch                                                   # PyTorch model device handling[6]
import soundfile as sf                                         # Audio file read/write[7]
import numpy as np                                             # Numeric array operations

from fastapi import FastAPI, UploadFile, File                  # API framework[8]
from scripts.preprocess_audio import preprocess_audio           # Resample & normalize audio[9]
from utils.audio_vad import split_audio                        # WebRTC VAD segmentation[10]
from utils.emotion_classifier import EmotionClassifier          # Hugging Face audio-classification pipeline[11]

# Correct import of Ultravox inference class
from ultravox.inference.ultravox_infer import UltravoxInference # Multi-modal STT+LLM pipeline[12]

# Kokoro TTS pipeline for expressive speech synthesis
from kokoro import KPipeline                                    # GPU-accelerated TTS[13]

# --- Load Configuration ---
config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)                                 # Contains model paths and settings[5]

# --- Set Device ---
device = torch.device(config.get("device", "cpu"))             # 'cuda' or 'cpu'[6]

# --- Initialize Models ---
# Instantiate UltravoxInference with correct constructor arguments
stt_llm = UltravoxInference(
    config["model"]["stt_llm_path"],                           # Path to Ultravox model directory
    device=device,                                             # Device for inference
    trust_remote_code=True                                     # Allow remote code execution
)

tts_model = KPipeline(lang_code='a')                           # 'a' for American English voice[13]

emotion_model = EmotionClassifier(
    config["emotion_classifier"]["model_name"],                # HF model ID for emotion detection
    device=config.get("device", "cpu")                         # Device for emotion pipeline
)

# --- FastAPI Application ---
app = FastAPI(title="Real-Time Emotional STS Agent")

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Processes uploaded audio:
      1. VAD-based segmentation
      2. Preprocess each segment (resample & normalize)
      3. STT + LLM transcription and reply generation
      4. Emotion classification
      5. TTS synthesis of reply
    Returns JSON with user_text, emotion, agent_text, and hex-encoded WAV.
    """
    # 1. Read and decode uploaded audio file
    data, sr = sf.read(io.BytesIO(await file.read()))          # Load audio into numpy array[7]

    # 2. Split into voiced segments
    segments = split_audio(data, sr, config["vad"]["aggressiveness"])  # VAD splits into speech chunks[10]

    results = []
    for seg in segments:
        # 3. Preprocess segment
        seg_norm = preprocess_audio(seg, sr)                    # Resample & normalize to 16kHz[9]

        # 4. STT → text transcription
        transcript = stt_llm({"audio": seg_norm, "sampling_rate": sr})[0]["text"]  # HF pipeline call[12]

        # 5. Emotion classification
        emotion = emotion_model.predict(seg_norm, sr)           # Returns label e.g., 'happiness'[11]

        # 6. LLM reply generation (emotion-conditioned)
        reply = stt_llm.generate_response(                      # Custom method on UltravoxInference
            prompt=transcript,
            emotion=emotion
        )

        # 7. TTS synthesis of agent reply
        tts_output = tts_model(reply, voice='af_heart')         # Returns list of (gen, pros, audio)[13]
        audio_array = tts_output[0][2]                          # Extract waveform array

        # 8. Encode WAV to hex string for JSON transport
        buf = io.BytesIO()
        sf.write(buf, audio_array, config.get("tts_sample_rate", 24000), format="WAV")
        hex_audio = buf.getvalue().hex()

        results.append({
            "user_text": transcript,
            "emotion": emotion,
            "agent_text": reply,
            "audio_hex": hex_audio
        })

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),                    # Use $PORT env or default 8000
        reload=True                                            # Auto-reload on file changes[8]
    )
