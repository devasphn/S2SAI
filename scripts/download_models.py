import torch
from huggingface_hub import snapshot_download
from transformers import pipeline

# Ultravox inference test
stt_llm = pipeline("audio-to-text", model="models/ultravox", device=0)
print("Ultravox loaded:", stt_llm)

# Kokoro TTS test (import from package)
from kokoro import KPipeline
tts = KPipeline(lang_code='a')
print("Kokoro loaded:", tts)

# Emotion classifier test
emo = pipeline("audio-classification", model="models/emotion", device=0)
print("Emotion model loaded:", emo)
