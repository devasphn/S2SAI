#!/usr/bin/env python3

from huggingface_hub import snapshot_download

def download_models():
    """
    Download required models from Hugging Face Hub into the local 'models/' directory.
    """

    # 1. Ultravox STT+LLM Model (v0.5 1B parameters)
    print("Downloading Ultravox v0.5 (1B) model...")
    snapshot_download(
        repo_id="fixie-ai/ultravox-v0_5-llama-3_2-1b",
        local_dir="models/ultravox",
        repo_type="model",
        resume_download=True
    )
    print("Ultravox model downloaded to 'models/ultravox'")

    # 2. Kokoro TTS Model
    print("Downloading Kokoro TTS model...")
    snapshot_download(
        repo_id="hexgrad/kokoro",
        local_dir="models/kokoro",
        repo_type="model",
        resume_download=True
    )
    print("Kokoro model downloaded to 'models/kokoro'")

    # 3. Emotion Classification Model
    print("Downloading audio emotion classifier...")
    snapshot_download(
        repo_id="LAION/EmoNet",
        local_dir="models/emotion_classifier",
        repo_type="model",
        resume_download=True
    )
    print("Emotion classifier model downloaded to 'models/emotion_classifier'")

if __name__ == "__main__":
    download_models()
