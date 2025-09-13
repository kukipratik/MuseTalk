import os
import numpy as np
import soundfile as sf
import onnxruntime as ort
from kokoro_onnx import Kokoro

MODEL_PATH = "models/kokoro/kokoro-v1.0.onnx"
VOICES_PATH = "models/kokoro/voices.json"
os.makedirs("kokoro_outputs", exist_ok=True)

# Show what ORT sees
avail = ort.get_available_providers()
print("Available providers:", avail)

# Prefer CUDA, then CPU; but only include ones that actually exist
preferred = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in avail]

# Build a session (try CUDA first, fallback to CPU)
so = ort.SessionOptions()
so.log_severity_level = 3  # quieter logs

session = None
errors = []
for ep in preferred:
    try:
        session = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=[ep])
        print(f"Using ONNX Runtime provider: {ep}")
        break
    except Exception as e:
        errors.append((ep, str(e)))

if session is None:
    # final safe fallback: pure CPU provider name in case list was empty
    session = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
    print("Fell back to CPUExecutionProvider")
    if errors:
        print("CUDA/TensorRT init errors (ignored):")
        for ep, msg in errors:
            print(f"  - {ep}: {msg.splitlines()[0]}")

# Create Kokoro normally, then inject our session
tts = Kokoro(MODEL_PATH, VOICES_PATH)
# kokoro_onnx exposes a .session attribute we can override
tts.session = session

text = "Quick GPU check. If CUDA is available I'll be fast; otherwise I'll run on CPU."
samples, sr = tts.create(text, voice="af_sarah", speed=1.0, lang="en-us")

print("dtype:", getattr(samples, "dtype", None), "shape:", np.array(samples).shape, "sr:", sr)
sf.write("kokoro_outputs/gpu_check.wav", samples, sr, subtype="PCM_16")
print("Saved `kokoro_outputs/gpu_check.wav`")
