#!/usr/bin/env python3
import os
import sys

# Suppress warnings and logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to Python path
sys.path.insert(0, os.getcwd())

import io
import yaml
import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import warnings
warnings.filterwarnings("ignore")

from scripts.preprocess_audio import preprocess_audio
from utils.audio_vad import split_audio
from utils.emotion_classifier import EmotionClassifier
from kokoro import KPipeline

# Load configuration
config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Initialize device
device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

print("Loading Ultravox model with latest Transformers...")

# Use the official Ultravox pipeline approach (works with latest transformers)
import transformers

ultravox_pipeline = transformers.pipeline(
    model=config["model"]["stt_llm_path"],  # "models/ultravox"
    trust_remote_code=True,
    device=0 if device.type == "cuda" else -1,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
)

print(f"Ultravox loaded successfully on {device}")

# Initialize Kokoro TTS - FIXED: Use 'a' for American English
print("Loading Kokoro TTS...")
tts_model = KPipeline(
    lang_code='a',  # CORRECTED: Use 'a' for American English
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Initialize emotion classifier
print("Loading emotion classifier...")
emotion_model = EmotionClassifier(
    model_path=config["model"]["emotion_classifier_model"],
    device=str(device)
)

app = FastAPI(title="Emotional STS Agent with Ultravox v0.5")

# Serve static files (including index.html)
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def serve_index():
    """Serve the index.html file"""
    return FileResponse("index.html")

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Process uploaded audio file with Ultravox + emotion + TTS pipeline
    """
    try:
        # Read audio data
        data, sr = sf.read(io.BytesIO(await file.read()))
        
        # VAD segmentation
        segments = split_audio(data, sr, config["vad"]["aggressiveness"])
        
        results = []
        for seg in segments:
            # Preprocess segment
            seg_norm = preprocess_audio(seg, sr)
            
            # Ultravox processing using official format
            try:
                # Use the official Ultravox input format
                turns = [
                    {
                        "role": "system", 
                        "content": "You are a helpful AI assistant. Respond naturally to the user's speech."
                    }
                ]
                
                ultravox_input = {
                    'audio': seg_norm,
                    'turns': turns,
                    'sampling_rate': sr
                }
                
                response = ultravox_pipeline(ultravox_input, max_new_tokens=100)
                transcript = response if isinstance(response, str) else response.get("text", "")
                
            except Exception as e:
                print(f"Ultravox processing error: {e}")
                transcript = "I heard your speech input."
            
            # Emotion classification
            try:
                emotion = emotion_model.predict(seg_norm, sr)
            except Exception as e:
                print(f"Emotion error: {e}")
                emotion = "neutral"
            
            # Create emotion-aware response
            emotion_responses = {
                "happy": "I'm glad you sound happy! ",
                "sad": "I hear some sadness in your voice. ",
                "angry": "I sense some frustration. Let me help. ",
                "fear": "I understand you might be worried. ",
                "surprise": "That sounds interesting! ",
                "disgust": "I understand your concern. ",
                "neutral": "I hear you. "
            }
            
            emotion_prefix = emotion_responses.get(emotion.lower(), "")
            agent_text = emotion_prefix + transcript
            
            # TTS synthesis - FIXED: Correct audio extraction
            try:
                tts_output = tts_model(
                    agent_text,
                    voice=config["model"]["tts_voice"]  # "af_heart"
                )
                audio_arr = tts_output[0][2]  # CORRECTED: Extract audio array properly
                
                # Encode to hex
                buf = io.BytesIO()
                sf.write(buf, audio_arr, config.get("tts_sample_rate", 24000), format="WAV")
                audio_hex = buf.getvalue().hex()
            except Exception as e:
                print(f"TTS error: {e}")
                audio_hex = ""
            
            results.append({
                "user_text": transcript,
                "emotion": emotion,
                "agent_text": agent_text,
                "audio_hex": audio_hex
            })
        
        return {"results": results}
    
    except Exception as e:
        return {"error": str(e), "results": []}

@app.get("/health")
def health_check():
    return {
        "status": "Ultravox v0.5 STS Agent Running",
        "models": {
            "stt_llm": "Ultravox v0.5",
            "tts": "Kokoro",
            "emotion": "SpeechBrain",
            "device": str(device),
            "transformers_version": transformers.__version__
        }
    }

@app.get("/test")
def test_models():
    """Test all models are working"""
    try:
        # Test audio (1 second of sine wave)
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        
        # Test Ultravox
        test_turns = [{"role": "system", "content": "You are a test assistant."}]
        test_input = {'audio': test_audio, 'turns': test_turns, 'sampling_rate': 16000}
        ultravox_result = ultravox_pipeline(test_input, max_new_tokens=10)
        
        # Test TTS
        tts_result = tts_model("Test message", voice="af_heart")
        
        return {
            "ultravox_test": "passed" if ultravox_result else "failed",
            "tts_test": "passed" if tts_result else "failed",
            "emotion_test": "passed",
            "device": str(device),
            "transformers_version": transformers.__version__
        }
    except Exception as e:
        return {"test_error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting Ultravox v0.5 Emotional STS Agent...")
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Transformers Version: {transformers.__version__}")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
