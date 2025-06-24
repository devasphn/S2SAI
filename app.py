import os
import io
import yaml
import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File
from scripts.preprocess_audio import preprocess_audio
from utils.audio_vad import split_audio
from utils.emotion_classifier import EmotionClassifier

# Hypothetical Ultravox and Kokoro imports
from ultravox import UltravoxForSpeech
from kokoro import KokoroTTSModel

# Load configuration
config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

device = torch.device(config.get("device", "cpu"))

# Initialize STT+LLM model
stt_llm = UltravoxForSpeech.from_pretrained(
    config["model"]["stt_llm_path"], device=device
)

# Initialize TTS model
tts_model = KokoroTTSModel.from_pretrained(
    config["model"]["tts_path"], device=device
)

# Initialize emotion classifier
emotion_model = EmotionClassifier(
    config["emotion_classifier"]["model_name"], config.get("device", "cpu")
)

app = FastAPI()

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    # Read and decode uploaded audio file
    data, sr = sf.read(io.BytesIO(await file.read()))
    # Split into voiced segments
    segments = split_audio(data, sr, config["vad"]["aggressiveness"])
    responses = []

    for seg in segments:
        # Preprocess segment
        seg_norm = preprocess_audio(seg, sr)

        # STT + LLM inference
        text = stt_llm.transcribe(seg_norm)

        # Emotion detection
        emotion = emotion_model.predict(seg_norm, sr)

        # TTS synthesis
        audio_out = tts_model.synthesize(text, emotion)

        responses.append({
            "text": text,
            "emotion": emotion,
            "audio_bytes": audio_out
        })

    return {"results": responses}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
