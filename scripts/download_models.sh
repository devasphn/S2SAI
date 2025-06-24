#!/usr/bin/env bash
set -e

# Clone Ultravox unified STT+LLM model
git clone https://github.com/fixie-ai/ultravox.git models/ultravox

# Clone Kokoro TTS model
git clone https://github.com/hexgrad/kokoro.git models/kokoro

# Clone Parler-TTS Expresso for emotional TTS
git clone https://github.com/huggingface/parler-tts.git models/parler-tts
