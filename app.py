#!/usr/bin/env python3
import os, sys

# Suppress TensorFlow logs and force PyTorch-only pipelines
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide TF INFO/WARNING/ERROR logs[7]
os.environ["TRANSFORMERS_NO_TF"]   = "1"  # Disable TensorFlow in Hugging Face Transformers[6]

# Ensure project root is on Python path for local package imports
sys.path.insert(0, os.getcwd())           # Include project root in module search[3]

import io
import yaml
import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, UploadFile, File  # FastAPI file uploads[1]

# Local utilities
from scripts.preprocess_audio import preprocess_audio  # Resample & normalize audio[4]
from utils.audio_vad import split_audio               # WebRTC VAD segmentation[5]
from utils.emotion_classifier import EmotionClassifier # SpeechBrain emotion detection[6]

# Ultravox unified STT+LLM inference
from ultravox.inference.ultravox_infer import UltravoxInference  # Ultravox API[9]

# Kokoro TTS pipeline for expressive speech synthesis
from kokoro import KPipeline  # GPU-accelerated TTS engine[2]

# Load configuration
config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)  # Parse YAML configuration[3]

# Select device for inference
device = torch.device(config.get("device", "cpu"))  # 'cuda' or 'cpu'[8]

# Initialize Ultravox STT+LLM model
stt_llm = UltravoxInference(
    config["model"]["stt_llm_path"],
    device=device
)

# Initialize Kokoro TTS model
tts_model = KPipeline(lang_code=config["model"]["tts_voice"])

# Initialize SpeechBrain emotion classifier
emotion_model = EmotionClassifier(
    model_path=config["model"]["emotion_classifier_model"],
    device=config.get("device", "cpu")
)

# Create FastAPI application
app = FastAPI(title="Emotional STS Agent with Kokoro TTS")

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Process uploaded audio file:
      1. Read and decode audio
      2. Segment speech via VAD
      3. Preprocess segments
      4. STT transcription and LLM reply
      5. Emotion classification
      6. TTS synthesis with Kokoro
      7. Return JSON with text, emotion, and hex-encoded WAV
    """
    # 1. Read audio data into numpy array
    data, sr = sf.read(io.BytesIO(await file.read()))  # PySoundFile I/O[4]

    # 2. VAD-based speech segmentation
    segments = split_audio(data, sr, config["vad"]["aggressiveness"])  # WebRTC VAD[5]

    results = []
    for seg in segments:
        # 3. Preprocess each segment (resample & normalize)
        seg_norm = preprocess_audio(seg, sr)  # Librosa-style processing[4]

        # 4. STT transcription and LLM reply generation
        transcript = stt_llm(
            {"audio": seg_norm, "sampling_rate": sr}
        )[0]["text"]  # Ultravox pipeline returns list of dicts[9]

        # 5. Emotion detection
        emotion = emotion_model.predict(seg_norm, sr)  # SpeechBrain inference[6]

        # 6. TTS synthesis of the LLM reply
        tts_output = tts_model(transcript)  # Returns list of (text, prosody, waveform)[2]
        audio_arr  = tts_output[0][2]       # Extract numpy waveform

        # 7. Encode WAV to hex string for JSON transport
        buf = io.BytesIO()
        sf.write(buf, audio_arr, config["tts_sample_rate"], format="WAV")
        hex_audio = buf.getvalue().hex()

        results.append({
            "user_text":  transcript,
            "emotion":    emotion,
            "agent_text": transcript,
            "audio_hex":  hex_audio
        })

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True  # Auto-reload on code changes[1]
    )
