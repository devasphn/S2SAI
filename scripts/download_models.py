#!/usr/bin/env python3
from huggingface_hub import snapshot_download

def download_models():
    """
    Download and cache all required models:
      1. Ultravox v0.5 (1B) for STT + LLM
      2. Kokoro-82M for Text-to-Speech
      3. LAION/EmoNet for audio emotion classification
    """

    # 1. Download Ultravox v0.5 1B model
    print("Downloading Ultravox v0.5 1B model...")
    snapshot_download(
        repo_id="fixie-ai/ultravox-v0_5-llama-3_2-1b",
        local_dir="models/ultravox",
        repo_type="model",
        resume_download=True
    )
    print("Ultravox model ready in 'models/ultravox'")

    # 2. Download Kokoro-82M TTS model
    print("Downloading Kokoro-82M TTS model...")
    snapshot_download(
        repo_id="hexgrad/Kokoro-82M",
        local_dir="models/kokoro",
        repo_type="model",
        resume_download=True
    )
    print("Kokoro-82M model ready in 'models/kokoro'")

    snapshot_download(
    repo_id="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    local_dir="models/emotion_classifier",
    repo_type="model",
    resume_download=True
    )
    print("Downloaded SpeechBrain emotion model to 'models/emotion_classifier'")

if __name__ == "__main__":
    download_models()
