from huggingface_hub import snapshot_download

# Download the latest Ultravox v0.5 1B model
snapshot_download(
    repo_id="fixie-ai/ultravox-v0_5-llama-3_2-1b",  # Correct repository ID
    local_dir="models/ultravox",
    repo_type="model",
    revision="main",
    resume_download=True
)

print("Ultravox model downloaded successfully!")
