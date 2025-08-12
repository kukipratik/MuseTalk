#!/usr/bin/env bash
set -euo pipefail

echo "[startup] begin"

# ---------- system deps ----------
apt-get update -y
apt-get install -y --no-install-recommends rsync ffmpeg
rm -rf /var/lib/apt/lists/*

# ---------- conda env ----------
if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda activate musetalk || { echo "[startup] conda env 'musetalk' missing"; exit 1; }
else
  echo "[startup] conda not found at /workspace/miniconda3"; exit 1
fi

# ---------- runtime env ----------
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_MODULE_LOADING=LAZY
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

SRC=/workspace/MuseTalk
DST=/root/musetalk_hot

mkdir -p "$DST"
echo "[startup] rsync -> $DST"
rsync -a --delete "$SRC/" "$DST/"

cd "$DST"
# SAFE expansion (fixes 'unbound variable')
export PYTHONPATH="$DST:${PYTHONPATH:-}"

# ---------- choose uvicorn app module ----------
if [ -f "$DST/api/server.py" ]; then
  UVICORN_APP="api.server:app"
elif [ -f "$DST/api_server.py" ]; then
  UVICORN_APP="api_server:app"
else
  echo "[startup] Could not find FastAPI app (api/server.py or api_server.py)."; exit 1
fi
export UVICORN_APP

# ---------- warm up (load models once) ----------
echo "[startup] warm-up: importing & loading weights"
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
pe = pe.half().to(device)
vae.vae = vae.vae.half().to(device)
unet.model = unet.model.half().to(device)
print("✅ Warm-up complete: models resident, first request will be hot.")
PY

# ---------- run server ----------
echo "[startup] starting uvicorn on :8000 -> $UVICORN_APP"
exec uvicorn "$UVICORN_APP" --host 0.0.0.0 --port 8000 --workers 1
