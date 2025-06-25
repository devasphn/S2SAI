#!/usr/bin/env python3
import os, sys

# Suppress TensorFlow logs and enforce PyTorch-only pipelines
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"]   = "1"

# Ensure local Ultravox package is importable
sys.path.insert(0, os.getcwd())

import yaml
import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, UploadFile, File

from scripts.preprocess_audio import preprocess_audio
from utils.audio_vad import split_audio
from utils.emotion_classifier import EmotionClassifier
from ultravox.inference.ultravox_infer import UltravoxInference
from kokoro import KPipeline

# Load configuration
config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Initialize models
device     = torch.device(config.get("device", "cpu"))
stt_llm    = UltravoxInference(config["model"]["stt_llm_path"], device=device)
tts_model  = KPipeline(lang_code=config["model"]["tts_voice"])
emotion_model = EmotionClassifier(
    model_path=config["model"]["emotion_classifier_model"],
    device=config.get("device", "cpu")
)

app = FastAPI()

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    data, sr = sf.read(io.BytesIO(await file.read()))
    segments = split_audio(data, sr, config["vad"]["aggressiveness"])
    results = []

    for seg in segments:
        seg_norm   = preprocess_audio(seg, sr)
        transcript = stt_llm({"audio": seg_norm, "sampling_rate": sr})[0]["text"]
        emotion    = emotion_model.predict(seg_norm, sr)
        reply      = stt_llm.generate_response(prompt=transcript, emotion=emotion)
        tts_out    = tts_model(reply)
        audio_arr  = tts_out[0][2]
        buf        = io.BytesIO()
        sf.write(buf, audio_arr, config["tts_sample_rate"], format="WAV")
        results.append({
            "user_text":  transcript,
            "emotion":    emotion,
            "agent_text": reply,
            "audio_hex":  buf.getvalue().hex()
        })

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT",8000)), reload=True)
