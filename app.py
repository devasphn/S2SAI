#!/usr/bin/env python3
import os
import sys

# 1. Suppress TensorFlow logs and force PyTorch-only in Transformers
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Suppress TF INFO/WARN/ERROR[1]
os.environ["TRANSFORMERS_NO_TF"]   = "1"      # Disable TensorFlow backend in HF Transformers[2]

# 2. Ensure project root is on Python path for local package imports
sys.path.insert(0, os.getcwd())               # Include project root in module search[3]

import io
import yaml
import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, UploadFile, File

# 3. Local utilities
from scripts.preprocess_audio import preprocess_audio  # Resample & normalize audio[4]
from utils.audio_vad import split_audio               # WebRTC VAD segmentation[5]
from utils.emotion_classifier import EmotionClassifier # HF audio-emotion pipeline[6]

# 4. Ultravox inference
from ultravox.inference.ultravox_infer import UltravoxInference  # STT + LLM[7]

# 5. Kokoro TTS
from kokoro import KPipeline                     # GPU-accelerated TTS[8]

# 6. Load configuration
config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)                   # Parse YAML config[9]

# 7. Set device
device = torch.device(config.get("device", "cpu"))  # cuda or cpu[10]

# 8. Initialize models
stt_llm       = UltravoxInference(config["model"]["stt_llm_path"], device=device)  # Load Ultravox[7]
tts_model     = KPipeline(lang_code='a')                       # 'a' = American English[8]
emotion_model = EmotionClassifier(                             
    config["emotion_classifier_model"], device=device         # Load emotion classifier[6]
)

# 9. FastAPI app
app = FastAPI(title="Emotional STS Agent")

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    # 9.1 Read and decode audio
    data, sr = sf.read(io.BytesIO(await file.read()))          # WAV → numpy[11]
    # 9.2 VAD segmentation
    segments = split_audio(data, sr, config["vad"]["aggressiveness"])  # Speech chunks[5]

    results = []
    for seg in segments:
        # 9.3 Preprocess
        seg_norm = preprocess_audio(seg, sr)                   # Resample & normalize[4]
        # 9.4 STT → text
        transcript = stt_llm({"audio": seg_norm, "sampling_rate": sr})[0]["text"]  # HF API[7]
        # 9.5 Emotion detection
        emotion = emotion_model.predict(seg_norm, sr)          # Label e.g., 'happy'[6]
        # 9.6 LLM reply conditioned on emotion
        reply = stt_llm.generate_response(prompt=transcript, emotion=emotion)  # Generate[7]
        # 9.7 TTS synthesis
        tts_output = tts_model(reply, voice='af_heart')        # Generate waveform[8]
        audio_arr  = tts_output[0][2]
        # 9.8 Encode WAV to hex for JSON
        buf = io.BytesIO()
        sf.write(buf, audio_arr, config.get("tts_sample_rate", 24000), format="WAV")
        hex_audio = buf.getvalue().hex()

        results.append({
            "user_text":  transcript,
            "emotion":    emotion,
            "agent_text": reply,
            "audio_hex":  hex_audio
        })

    return {"results": results}                                # Return JSON[9]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True
    )                                                             # Auto-reload[12]
