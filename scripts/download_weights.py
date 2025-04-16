from huggingface_hub import snapshot_download

repo_id = "armeet/nomri"
repo_path = snapshot_download(repo_id, local_dir="weights")

print(f"Downloaded to: {repo_path}")
