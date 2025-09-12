#!/bin/bash
set -euo pipefail

# === Settings ===
CheckpointsDir="models/kokoro"

# === Helpers ===
download_with_resume () {
  local url="$1"
  local out="$2"
  # -L follow redirects, -C - resume, --fail error on 4xx/5xx
  curl -L --fail -C - -o "$out" "$url"
}

download_if_missing () {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "✔ Skipping (exists): $out"
  else
    echo "↓ Downloading: $(basename "$out")"
    download_with_resume "$url" "$out"
    echo "✔ Done: $out"
  fi
}

# === Create directories ===
mkdir -p \
  "$CheckpointsDir"

# === Tools ===
pip install -U "huggingface_hub[cli]"
pip install gdown

# Mirror endpoint for HF (optional)
export HF_ENDPOINT=https://hf-mirror.com

# === Kokoro ONNX + Voices ===
download_if_missing \
  "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" \
  "$CheckpointsDir/kokoro-v1.0.onnx"

download_if_missing \
  "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" \
  "$CheckpointsDir/voices-v1.0.bin"

echo "✅ All weights have been downloaded successfully!"
