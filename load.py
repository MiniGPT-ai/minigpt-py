import glob
import os, requests, zipfile
import torch
from .config import MiniGPTConfig
from .model import MiniGPTModel

BASE_URL = "https://github.com/MiniGPT-ai/MiniGPT/releases/download"
CACHE_DIR = os.path.expanduser("~/.cache/minigpt")

MODEL_VERSIONS = {
    "minigpt0-preview": {
        "tag": "v0.1.0-preview",
        "zip": "Minigpt-0-preview.zip"
    },
    "minigpt0": {
        "tag": "v0.1.0",
        "zip": "Minigpt-0.zip"
    }
}


def download_and_extract_zip(url, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    zip_path = os.path.join(extract_to, "model.zip")

    if not os.path.exists(os.path.join(extract_to, "minigpt0.pth")):
        print(f"⬇️ Downloading: {url}")
        r = requests.get(url)
        r.raise_for_status()

        with open(zip_path, "wb") as f:
            f.write(r.content)

        print("📦 Extracting model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        os.remove(zip_path)


def load_model(version="minigpt0", device="cpu"):
    if version not in MODEL_VERSIONS:
        raise ValueError(f"❌ Unknown model version: {version}")

    meta = MODEL_VERSIONS[version]
    tag = meta["tag"]
    zip_name = meta["zip"]

    target_dir = os.path.join(CACHE_DIR, version)
    zip_url = f"{BASE_URL}/{tag}/{zip_name}"

    download_and_extract_zip(zip_url, target_dir)

    # Load config
    config_path = os.path.join(target_dir, "config.json")
    config = MiniGPTConfig(config_path)

    # 🔍 Auto-detect the .pth weight file
    weight_files = glob.glob(os.path.join(target_dir, "*.pth"))
    if not weight_files:
        raise FileNotFoundError("❌ No .pth model file found after extraction.")
    weights_path = weight_files[0]

    # Load model
    model = MiniGPTModel(config)
    model.config = config  # optional
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # Tokenizer files
    vocab = os.path.join(target_dir, "tokenizer", "vocab.json")
    merges = os.path.join(target_dir, "tokenizer", "merges.txt")

    return model, vocab, merges
