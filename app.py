#!/usr/bin/env python3
import os, sys

# Suppress TensorFlow logs and enforce PyTorch-only pipelines
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"]   = "1"

# Ensure local packages are importable
sys.path.insert(0, os.getcwd())

import io
import yaml
import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, UploadFile, File

# Import Ultravox custom classes
sys.path.append("models/ultravox")
from ultravox_pipeline import UltravoxPipeline
from ultravox_config import UltravoxConfig
from ultravox_model import UltravoxModel

from scripts.preprocess_audio import preprocess_audio
from utils.audio_vad import split_audio
from utils.emotion_classifier import EmotionClassifier
from kokoro import KPipeline

# Load configuration
config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Initialize models
device = torch.device(config.get("device", "cpu"))

# Load Ultravox using its custom pipeline
stt_llm = UltravoxPipeline.from_pretrained(
    config["model"]["stt_llm_path"],
    device_map="auto" if device.type == "cuda" else None,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
)

# Initialize Kokoro TTS with local model path
tts_model = KPipeline(
    model_path=config["model"]["kokoro_model_path"],
    lang_code='a'
)

# Initialize emotion classifier
emotion_model = EmotionClassifier(
    model_path=config["model"]["emotion_classifier_model"],
    device=config.get("device", "cpu")
)

app = FastAPI(title="Emotional STS Agent")

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Process uploaded audio file:
    1. VAD segmentation
    2. STT transcription with Ultravox
    3. Emotion classification  
    4. Generate agent response
    5. TTS synthesis
    6. Return JSON with text, emotion, and hex-encoded audio
    """
    # Read audio data
    data, sr = sf.read(io.BytesIO(await file.read()))
    
    # VAD segmentation
    segments = split_audio(data, sr, config["vad"]["aggressiveness"])
    results = []

    for seg in segments:
        # Preprocess segment
        seg_norm = preprocess_audio(seg, sr)
        
        # STT transcription using Ultravox custom pipeline
        # Ultravox expects specific input format
        inputs = {
            "audio": seg_norm,
            "text": "",  # Empty text for pure ASR
            "sampling_rate": sr
        }
        
        transcript_result = stt_llm(inputs)
        transcript = transcript_result["text"] if isinstance(transcript_result, dict) else str(transcript_result)
        
        # Emotion classification
        emotion = emotion_model.predict(seg_norm, sr)
        
        # Generate agent response (you can enhance this with LLM integration)
        agent_text = f"I understand you said: {transcript}. I detect you're feeling {emotion}."
        
        # TTS synthesis with specified voice
        tts_output = tts_model(
            agent_text, 
            voice=config["model"]["tts_voice"]
        )
        audio_arr = tts_output[0][2]
        
        # Encode audio to hex for JSON transport
        buf = io.BytesIO()
        sf.write(buf, audio_arr, config.get("tts_sample_rate", 24000), format="WAV")
        
        results.append({
            "user_text": transcript,
            "emotion": emotion,
            "agent_text": agent_text,
            "audio_hex": buf.getvalue().hex()
        })

    return {"results": results}

@app.get("/")
def health_check():
    return {"status": "Emotional STS Agent is running", "models_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
