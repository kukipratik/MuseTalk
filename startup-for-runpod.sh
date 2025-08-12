#!/usr/bin/env bash
set -euo pipefail

export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_MODULE_LOADING=LAZY
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Copy code/models from network volume to local container storage (faster I/O)
SRC=/workspace/MuseTalk
DST=/root/musetalk_hot
mkdir -p "$DST"
rsync -a --delete "$SRC/" "$DST/"
cd "$DST"

# Warm-up: load models into GPU once so first request is instant
python - <<'PY'
import torch
from musetalk.utils.utils import load_all_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vae, unet, pe = load_all_model(
    unet_model_path="models/musetalkV15/unet.pth",
    vae_type="sd-vae",
    unet_config="models/musetalkV15/musetalk.json",
    device=device,
    vae_dir="models/sd-vae",
)
print("✅ Warm-up complete: models loaded in VRAM.")
PY

# Start FastAPI app with uvicorn (keeps container hot)
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
