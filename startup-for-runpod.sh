#!/usr/bin/env bash
set -euo pipefail

echo "[startup] begin"

# ---------- system deps ----------
apt-get update -y
apt-get install -y --no-install-recommends ffmpeg
rm -rf /var/lib/apt/lists/*

# ---------- conda env ----------
if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /workspace/conda/etc/profile.d/conda.sh
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
# predictable CPU usage (tune if you do heavy CPU work)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Optional: faster temp I/O if you write frames; increase shm size on your pod if needed.
export TMPDIR=/dev/shm
mkdir -p /dev/shm/musetalk_tmp || true

# ---------- app dir ----------
cd /workspace/MuseTalk
export PYTHONPATH="/workspace/MuseTalk:${PYTHONPATH:-}"

# ---------- choose uvicorn app ----------
if [ -f "api/server.py" ]; then
  UVICORN_APP="api.server:app"
elif [ -f "api_server.py" ]; then
  UVICORN_APP="api_server:app"
else
  echo "[startup] Could not find FastAPI app (api/server.py or api_server.py)."; exit 1
fi
export UVICORN_APP

# ---------- run server (warm-up happens inside FastAPI startup) ----------
echo "[startup] starting uvicorn on :8000 -> $UVICORN_APP"
exec uvicorn "$UVICORN_APP" --host 0.0.0.0 --port 8000 --workers 1
# Tip: if you have uvloop/httptools installed, you can add:
#   --loop uvloop --http httptools
