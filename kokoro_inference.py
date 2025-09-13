import os, numpy as np, soundfile as sf
import onnxruntime as ort
from kokoro_onnx import Kokoro

print("Available providers:", ort.get_available_providers())

MODEL_PATH = "models/kokoro/kokoro-v1.0.onnx"
VOICES_PATH = "models/kokoro/voices.json"

# reduce CPU spin + threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

# build a CUDA-only session (no TensorRT)
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1

sess = ort.InferenceSession(
    MODEL_PATH,
    sess_options=so,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

# pass the prebuilt session into Kokoro (supported in 0.3.3)
tts = Kokoro(MODEL_PATH, VOICES_PATH, session=sess)

# --- warmup (tiny text) to JIT numba & ORT kernels ---
_ = tts.create("warmup", voice="af_sarah", speed=1.0, lang="en-us")

# real run
text = "GPU check run, hopefully fast and quiet on the CPU."
samples, sr = tts.create(text, voice="af_sarah", speed=1.0, lang="en-us")
print("dtype:", getattr(samples, "dtype", None), "len:", len(np.array(samples)), "sr:", sr)
sf.write("kokoro_outputs/gpu_check.wav", samples, sr, subtype="PCM_16")
print("Saved kokoro_outputs/gpu_check.wav")
