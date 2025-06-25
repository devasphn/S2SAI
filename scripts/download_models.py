from huggingface_hub import snapshot_download

# Download Ultravox v0.5 8B model
snapshot_download(
    repo_id="fixie-ai/ultravox-v0_5-llama-3_1-8b",
    local_dir="models/ultravox",
    repo_type="model",
    revision="main",
    resume_download=True
)

print("Ultravox 8B model downloaded successfully!")
