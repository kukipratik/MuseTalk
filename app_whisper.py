import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# ---------------- Config ----------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v2")   # you can change: tiny.en / base.en / small.en / medium.en / large-v2
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")      # "cuda" if you want GPU
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # "float16" if GPU, "int8" if CPU

# ---------------- Init ----------------
print(f"[whisper] Loading model '{WHISPER_MODEL}' on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})")
model = WhisperModel(
    WHISPER_MODEL,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)

app = FastAPI(title="Whisper STT Test API", version="1.0")


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Upload audio file -> returns transcribed text"""
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # save to tmp
    try:
        suffix = os.path.splitext(audio.filename)[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to buffer upload: {e}")

    # run inference
    try:
        segments, info = model.transcribe(
            tmp_path,
            beam_size=1,
            vad_filter=True,
        )
        full_text = "".join([seg.text for seg in segments]).strip()
        return JSONResponse({
            "text": full_text,
            "language": info.language,
            "language_probability": info.language_probability
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


@app.get("/health")
async def health():
    return {"status": "ok", "device": WHISPER_DEVICE, "model": WHISPER_MODEL}
