#!/usr/bin/env python3
import os
# Suppress TensorFlow logs and disable TF backend
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"]   = "1"

import io
import yaml
import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, UploadFile, File
from scripts.preprocess_audio import preprocess_audio
from utils.audio_vad import split_audio
from utils.emotion_classifier import EmotionClassifier

# Correct Ultravox import
from ultravox.inference.ultravox_infer import UltravoxInference

# Kokoro TTS pipeline
from kokoro import KPipeline

# Load configuration
with open(os.getenv("CONFIG_PATH", "config/settings.yaml")) as f:
    config = yaml.safe_load(f)

device = torch.device(config.get("device", "cpu"))

# Initialize models
stt_llm       = UltravoxInference.from_pretrained(
                   config["model"]["stt_llm_path"],
                   device=device,
                   trust_remote_code=True
               )
tts_model     = KPipeline(lang_code='a')
emotion_model = EmotionClassifier(
                   config["emotion_classifier"]["model_name"],
                   config.get("device", "cpu")
               )

app = FastAPI()

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    data, sr = sf.read(io.BytesIO(await file.read()))
    segments = split_audio(data, sr, config["vad"]["aggressiveness"])
    results = []

    for seg in segments:
        seg_norm    = preprocess_audio(seg, sr)
        user_text   = stt_llm({'audio': seg_norm, 'sampling_rate': sr})[0]['text']
        emotion     = emotion_model.predict(seg_norm, sr)
        agent_text  = stt_llm.generate_response(prompt=user_text, emotion=emotion)
        audio_out   = tts_model(agent_text, voice='af_heart')[0][2]
        buf         = io.BytesIO()
        sf.write(buf, audio_out, config.get("tts_sample_rate", 24000), format="WAV")
        results.append({
            "user_text":  user_text,
            "emotion":    emotion,
            "agent_text": agent_text,
            "audio_hex":  buf.getvalue().hex()
        })

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
