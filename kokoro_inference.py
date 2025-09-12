import os
import soundfile as sf
from kokoro_onnx import Kokoro
import onnxruntime as ort
import numpy as np

print("Providers:", ort.get_available_providers())

MODEL_PATH = "models/kokoro/kokoro-v1.0.onnx"
VOICES_PATH = "models/kokoro/voices-v1.0.bin"

# 1) make sure the output folder exists
os.makedirs("kokoro_outputs", exist_ok=True)

tts = Kokoro(MODEL_PATH, VOICES_PATH)

text = "GPU check: this should use CUDA if available. Bro, i am good man. I like playing football; i am great lol. Hahaha just kidding hahahaha. Bye bye."
samples, sr = tts.create(text, voice="af_sarah", speed=1.0, lang="en-us")

# (optional) sanity prints
print("dtype:", getattr(samples, "dtype", None), "shape:", np.array(samples).shape, "sr:", sr)

# 2) write as 16-bit PCM (some players like this better on Windows)
sf.write("kokoro_outputs/gpu_check.wav", samples, sr, subtype="PCM_16")

print("Saved `kokoro_outputs/gpu_check.wav`")
