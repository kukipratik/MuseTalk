import os
import threading
from typing import Optional
from faster_whisper import WhisperModel

# --- Read env or defaults ---
WHISPER_MODEL = "tiny.en"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

print(f"[whisper_runtime] Loading faster-whisper '{WHISPER_MODEL}' on '{WHISPER_DEVICE}' ({WHISPER_COMPUTE_TYPE})")

# --- Singleton holder ---
_model: Optional[WhisperModel] = None
_lock = threading.Lock()

def get_stt_model() -> WhisperModel:
    global _model
    if _model is None:
        with _lock:
            if _model is None:  # double-checked locking
                # print(f"[whisper_runtime] Loading faster-whisper '{WHISPER_MODEL}' on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})")
                _model = WhisperModel(
                    WHISPER_MODEL,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                )
    return _model
