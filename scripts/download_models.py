from huggingface_hub import snapshot_download

# Download the Ultravox repo with weights into 'models/ultravox'
snapshot_download(
    repo_id="fixie-ai/ultravox",
    cache_dir="models/ultravox",
    resume_download=True
)
