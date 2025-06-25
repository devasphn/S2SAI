#!/usr/bin/env python3
import os
import sys

# 1) Suppress TensorFlow logs and force PyTorch-only pipelines
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"]   = "1"

# 2) Ensure project root is on Python path for local imports
sys.path.insert(0, os.getcwd())

import io
import yaml
import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, UploadFile, File
from transformers import pipeline

from scripts.preprocess_audio import preprocess_audio
from utils.audio_vad import split_audio
from utils.emotion_classifier import EmotionClassifier
from kokoro import KPipeline

# 3) Load your YAML configuration
config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 4) Determine device index (0 for GPU, -1 for CPU)
device_index = 0 if torch.cuda.is_available() else -1

# 5) Initialize Ultravox with meta-tensor disabled
ultravox_pipeline = pipeline(
    model=config["model"]["stt_llm_path"],       # e.g. "models/ultravox"
    trust_remote_code=True,
    device=device_index,
    torch_dtype=torch.float16 if device_index >= 0 else torch.float32,
    device_map=None,              # ← disable automatic meta-tensor device mapping
    low_cpu_mem_usage=False       # ← turn off meta-tensor init
)

# 6) Initialize Kokoro TTS
tts_model = KPipeline(
    model_path=config["model"]["kokoro_model_path"],  # "models/kokoro"
    lang_code='a'                                     # American English
)

# 7) Initialize SpeechBrain emotion classifier
emotion_model = EmotionClassifier(
    model_path=config["model"]["emotion_classifier_model"],  # "models/emotion_classifier"
    device=config.get("device", "cpu")
)

# 8) Create FastAPI app
app = FastAPI(title="Emotional STS Agent")

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    """
    1. Load audio into numpy array
    2. Split via VAD
    3. Preprocess each segment
    4. Transcribe + LLM with Ultravox
    5. Classify emotion
    6. Synthesize speech with Kokoro
    7. Return JSON with transcript, emotion, and WAV hex
    """
    # a) Read raw audio
    data, sr = sf.read(io.BytesIO(await file.read()))

    # b) VAD segmentation
    segments = split_audio(data, sr, config["vad"]["aggressiveness"])

    results = []
    for seg in segments:
        # c) Preprocess (resample/normalize)
        seg_norm = preprocess_audio(seg, sr)

        # d) STT → transcript + LLM response
        try:
            # Ultravox expects a "turns" structure
            ultr_input = {
                "audio": seg_norm,
                "turns": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": ""}
                ],
                "sampling_rate": sr
            }
            reply = ultravox_pipeline(ultr_input, max_new_tokens=50)
            transcript = reply if isinstance(reply, str) else reply.get("text", "")
        except Exception as e:
            print(f"Ultravox error: {e}")
            transcript = ""

        # e) Emotion classification
        try:
            emotion = emotion_model.predict(seg_norm, sr)
        except Exception as e:
            print(f"Emotion error: {e}")
            emotion = "neutral"

        # f) Generate a simple echo reply if needed
        if not transcript:
            agent_text = "Sorry, I couldn't transcribe that."
        else:
            agent_text = f"I heard: {transcript}. I detect {emotion} emotion."

        # g) TTS synthesis
        try:
            tts_out = tts_model(agent_text, voice=config["model"]["tts_voice"])
            audio_arr = tts_out[0][2]
            buf = io.BytesIO()
            sf.write(buf, audio_arr, config.get("tts_sample_rate", 24000), format="WAV")
            audio_hex = buf.getvalue().hex()
        except Exception as e:
            print(f"TTS error: {e}")
            audio_hex = ""

        results.append({
            "user_text":  transcript,
            "emotion":    emotion,
            "agent_text": agent_text,
            "audio_hex":  audio_hex
        })

    return {"results": results}

@app.get("/")
def health_check():
    return {"status": "Agent running", "models_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
