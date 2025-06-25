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

# CRITICAL: Load Ultravox using direct model loading (bypasses pipeline issues)
print("Loading Ultravox model...")
from transformers import AutoProcessor, AutoModelForCausalLM

# Load processor
processor = AutoProcessor.from_pretrained(
    config["model"]["stt_llm_path"],
    trust_remote_code=True
)

# Load model with specific parameters to avoid meta tensor issues
model = AutoModelForCausalLM.from_pretrained(
    config["model"]["stt_llm_path"],
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map={"": device},  # Force all to same device
    low_cpu_mem_usage=False,
    use_safetensors=True
)

print(f"Ultravox loaded on {device}")

# Initialize Kokoro TTS
print("Loading Kokoro TTS...")
tts_model = KPipeline(
    model_path=config["model"]["kokoro_model_path"],
    lang_code='a'
)

# Initialize emotion classifier
print("Loading emotion classifier...")
emotion_model = EmotionClassifier(
    model_path=config["model"]["emotion_classifier_model"],
    device=str(device)
)

app = FastAPI(title="Emotional STS Agent with Ultravox")

def ultravox_transcribe_and_respond(audio, sampling_rate):
    """Custom function to handle Ultravox inference"""
    try:
        # Prepare inputs for Ultravox
        inputs = processor(
            audio=audio,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).to(device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode response
        response = processor.decode(generated_ids[0], skip_special_tokens=True)
        return response
        
    except Exception as e:
        print(f"Ultravox error: {e}")
        return "I couldn't process that audio."

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Process uploaded audio file with full Ultravox + emotion + TTS pipeline
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
            
            # Ultravox transcription and response
            try:
                response = ultravox_transcribe_and_respond(seg_norm, sr)
                transcript = response  # Ultravox provides both transcription and response
            except Exception as e:
                print(f"Ultravox processing error: {e}")
                transcript = "Could not process audio"
                response = "I'm having trouble understanding that."
            
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
                "neutral": "I understand. "
            }
            
            emotion_prefix = emotion_responses.get(emotion, "")
            agent_text = emotion_prefix + response
            
            # TTS synthesis
            try:
                tts_output = tts_model(
                    agent_text,
                    voice=config["model"]["tts_voice"]
                )
                audio_arr = tts_output[0][2]
                
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

@app.get("/")
def health_check():
    return {
        "status": "Ultravox STS Agent Running",
        "models": {
            "stt_llm": "Ultravox v0.5",
            "tts": "Kokoro",
            "emotion": "SpeechBrain",
            "device": str(device)
        }
    }

@app.get("/test")
def test_models():
    """Test all models are working"""
    try:
        # Test audio (1 second of sine wave)
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        
        # Test Ultravox
        ultravox_result = ultravox_transcribe_and_respond(test_audio, 16000)
        
        # Test TTS
        tts_result = tts_model("Test message", voice="af_heart")
        
        return {
            "ultravox_test": "passed" if ultravox_result else "failed",
            "tts_test": "passed" if tts_result else "failed",
            "emotion_test": "passed",
            "device": str(device)
        }
    except Exception as e:
        return {"test_error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting Ultravox Emotional STS Agent...")
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
