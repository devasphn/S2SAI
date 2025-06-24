#!/usr/bin/env python3
import os                                                        # Manage OS environment variables[1]
# Silence TensorFlow logs and disable TF backend in Transformers
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"                         # Suppress TF INFO/WARNING/ERROR[2]
os.environ["TRANSFORMERS_NO_TF"]   = "1"                         # Force PyTorch-only in HF Transformers[3]

import io                                                        # Handle in-memory byte streams[4]
import yaml                                                      # Parse YAML configuration[5]
import torch                                                     # PyTorch for model devices[6]
import soundfile as sf                                           # Read/write audio files[7]
import numpy as np                                               # Numeric arrays for audio ops
from fastapi import FastAPI, UploadFile, File                    # Web framework for API endpoints[8]

# Audio preprocessing and VAD utilities
from scripts.preprocess_audio import preprocess_audio             # Resample & normalize audio[9]
from utils.audio_vad import split_audio                          # WebRTC VAD segmentation[10]
from utils.emotion_classifier import EmotionClassifier            # HF audio-classification pipeline[3]

# Ultravox unified STT + LLM inference
from ultravox.inference.ultravox_infer import UltravoxInference   # Load UltravoxInference class

# Kokoro TTS pipeline for speech synthesis
from kokoro import KPipeline                                      # Real-time TTS generator

# --- Load Configuration ---
config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")
with open(config_path, "r") as cfg:
    config = yaml.safe_load(cfg)                                 # Load model paths & settings[5]

# --- Set Device ---
device = torch.device(config.get("device", "cpu"))               # 'cuda' or 'cpu'[6]

# --- Initialize Models ---
stt_llm       = UltravoxInference(                               # STT → LLM conversational agent
                   model_name_or_path=config["model"]["stt_llm_path"],
                   device=device,
                   trust_remote_code=True
               )
tts_model     = KPipeline(lang_code='a')                        # 'a' = American English voice
emotion_model = EmotionClassifier(                               # Emotion detection pipeline
                   model_name=config["emotion_classifier"]["model_name"],
                   device=config.get("device", "cpu")
               )

# --- FastAPI Application ---
app = FastAPI(title="Real-Time Emotional STS Agent")

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file upload, performs VAD-based segmentation,
    runs STT+LLM inference with emotional context, and returns
    synthesized speech in hex-encoded WAV format.
    """
    # 1. Read uploaded audio
    data, sr = sf.read(io.BytesIO(await file.read()))            # Load WAV/FLAC input into numpy array[7]

    # 2. Split into voiced segments
    segments = split_audio(data, sr, config["vad"]["aggressiveness"])  # Frame-level VAD segmentation[10]

    results = []
    for seg in segments:
        # 3. Preprocess segment
        seg_norm = preprocess_audio(seg, sr)                      # Resample to 16 kHz & normalize amplitude[9]

        # 4. STT → Text transcription
        transcript = stt_llm({"audio": seg_norm, "sampling_rate": sr})[0]["text"]  # HF-style API[3]

        # 5. Emotion Classification
        emotion = emotion_model.predict(seg_norm, sr)            # Returns label e.g., 'happiness'[3]

        # 6. LLM Reply Generation (emotion-conditioned)
        reply = stt_llm.generate_response(prompt=transcript, emotion=emotion)  # Custom method on UltravoxInference

        # 7. TTS Synthesis
        tts_output = tts_model(reply, voice='af_heart')          # Returns list of (gen, pros, audio_array)
        audio_array = tts_output[0][2]                           # Extract numpy waveform

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
        port=int(os.getenv("PORT", 8000)),
        reload=True                                              # Auto-reload on file changes[8]
    )
