#!/usr/bin/env python3
import os                                                      # Manage environment variables[1]
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"                       # Suppress TensorFlow logs[2]
os.environ["TRANSFORMERS_NO_TF"]   = "1"                       # Force PyTorch-only pipelines[3]

import io                                                      # In-memory byte streams[4]
import yaml                                                    # YAML parsing[5]
import torch                                                   # Device handling[6]
import soundfile as sf                                         # Audio I/O[7]
import numpy as np                                             # Numeric operations

from fastapi import FastAPI, UploadFile, File                  # Web framework[8]
from scripts.preprocess_audio import preprocess_audio           # Resample & normalize audio[9]
from utils.audio_vad import split_audio                        # VAD segmentation[10]
from utils.emotion_classifier import EmotionClassifier          # Emotion classification[11]

# Correct Ultravox import
from ultravox.inference.ultravox_infer import UltravoxInference # Ultravox STT+LLM pipeline[11]

# Kokoro TTS pipeline
from kokoro import KPipeline                                    # GPU-accelerated TTS[13]

# Load configuration
with open(os.getenv("CONFIG_PATH", "config/settings.yaml")) as f:
    config = yaml.safe_load(f)                                 # Model paths & settings[5]

device = torch.device(config.get("device", "cpu"))             # 'cuda' or 'cpu'[6]

# Initialize models (removed unsupported 'trust_remote_code')
stt_llm       = UltravoxInference(
                   config["model"]["stt_llm_path"],
                   device=device
               )
tts_model     = KPipeline(lang_code='a')                       # 'a' = American English[13]
emotion_model = EmotionClassifier(
                   config["emotion_classifier"]["model_name"],
                   device=config.get("device", "cpu")
               )

app = FastAPI(title="Real-Time Emotional STS Agent")

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    data, sr = sf.read(io.BytesIO(await file.read()))          # Load audio[7]
    segments = split_audio(data, sr, config["vad"]["aggressiveness"])  # VAD splits[10]

    results = []
    for seg in segments:
        seg_norm   = preprocess_audio(seg, sr)                  # Resample & normalize[9]
        transcript = stt_llm({"audio": seg_norm, "sampling_rate": sr})[0]["text"]  # STT+LLM[11]
        emotion    = emotion_model.predict(seg_norm, sr)        # Emotion label[11]
        reply      = stt_llm.generate_response(prompt=transcript, emotion=emotion)  # LLM reply

        tts_output = tts_model(reply, voice='af_heart')         # TTS generation[13]
        audio_arr  = tts_output[0][2]
        buf        = io.BytesIO()
        sf.write(buf, audio_arr, config.get("tts_sample_rate", 24000), format="WAV")
        results.append({
            "user_text":  transcript,
            "emotion":    emotion,
            "agent_text": reply,
            "audio_hex":  buf.getvalue().hex()
        })

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),                    # Default port 8000
        reload=True                                            # Auto-reload on changes[8]
    )
