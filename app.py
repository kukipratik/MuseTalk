# ---------------- env must be set BEFORE heavy imports ----------------
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# ---------------- std imports ----------------
import time
import shutil
import tempfile
from typing import Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from transformers import WhisperModel

# ---------------- project imports ----------------
from musetalk.utils.utils import load_all_model
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor

from scripts.api_inference import inject_runtime, Avatar

import threading
_PIPELINE_LOCK = threading.Lock()

# ---------------- config (adjust paths if needed) ----------------
UNET_MODEL_PATH = "models/musetalkV15/unet.pth"
UNET_CONFIG_PATH = "models/musetalkV15/musetalk.json"
VAE_DIR          = "models/sd-vae"          # contains diffusion_pytorch_model.bin
WHISPER_DIR      = "models/whisper"
VERSION          = "v15"
DEFAULT_BATCH    = 20


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm models into THIS uvicorn worker; keep them resident for requests."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # perf knobs
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # --- load nets ---
    vae, unet, pe = load_all_model(
        unet_model_path=UNET_MODEL_PATH,
        vae_type="sd-vae",
        unet_config=UNET_CONFIG_PATH,
        device=device,
    )
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    # audio + whisper
    audio_processor = AudioProcessor(feature_extractor_path=WHISPER_DIR)
    whisper = WhisperModel.from_pretrained(WHISPER_DIR, low_cpu_mem_usage=True)
    whisper = whisper.to(device=device, dtype=torch.float16).eval()
    whisper.requires_grad_(False)

    # face parsing (used during avatar preparation)
    fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

    # constant timestep tensor
    timesteps = torch.tensor([0], device=device)

    # hand over to realtime module (so Avatar reuses hot models)
    inject_runtime(
        device=device,
        vae=vae,
        unet=unet,
        pe=pe,
        whisper=whisper,
        audio_processor=audio_processor,
        timesteps=timesteps,
        fp=fp,
        weight_dtype=unet.model.dtype,
    )

    # tiny CUDA prime to JIT kernels
    x = torch.randn(1, 3, 64, 64, device=device, dtype=torch.float16)
    _ = torch.nn.functional.conv2d(x, torch.randn(8, 3, 3, 3, device=device, dtype=torch.float16))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("FastAPI warm-up complete: models in VRAM and ready.")

    # make them available via app.state if you ever need (optional)
    app.state.device = device

    # startup done
    yield

    # --- shutdown ---
    print("Shutting down FastAPI server, cleaning up...")
    torch.cuda.empty_cache()


app = FastAPI(title="MuseTalk Realtime API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "cuda": torch.cuda.is_available()}


@app.post("/infer")
async def infer(
    # required form fields
    avatar_id: str = Form(...),
    # typical runtime controls
    fps: int = Form(25),
    bbox_shift: int = Form(5),
    batch_size: int = Form(DEFAULT_BATCH),
    preparation: bool = Form(False),
    # if you need to create an avatar the first time, send video_path
    video_path: Optional[str] = Form(None),
    # audio file (prefer 16kHz mono PCM .wav)
    audio: UploadFile = File(...),
):
    """
    Run MuseTalk inference using hot models.
    - If `preparation=False`, the avatar must exist under results/{v}/avatars/{avatar_id}/
    - If `preparation=True`, provide `video_path` (mp4 or folder with .png frames) to (re)create the avatar cache.
    """
    print(f"- infer called: avatar={avatar_id} fps={fps} prep={preparation} video_path={video_path}")
    t0 = time.perf_counter()

    # save audio to a temp file
    with tempfile.TemporaryDirectory() as tmpd:
        audio_path = os.path.join(tmpd, audio.filename or "audio.wav")
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        if preparation and not video_path:
            raise HTTPException(status_code=400, detail="video_path is required when preparation=True")

        # inject argparse-like namespace used by Avatar (since we're not running CLI)
        import types, scripts.api_inference as rt

        with _PIPELINE_LOCK:
            if not hasattr(rt, "args"):
                rt.args = types.SimpleNamespace(
                    version=VERSION,
                    extra_margin=10,
                    parsing_mode="jaw",
                    audio_padding_length_left=2,
                    audio_padding_length_right=2,
                    skip_save_images=False,  # saving frames+mp4 by default
                )

            # Create Avatar (reuses injected runtime; NO model reloads)
            try:
                avatar = Avatar(
                    avatar_id=avatar_id,
                    video_path=video_path or f"data/video/{avatar_id}.mp4",
                    bbox_shift=bbox_shift,
                    batch_size=batch_size,
                    preparation=preparation,
                )
            except SystemExit as e:
                # Avatar may sys.exit(1) if cache missing and not preparing
                raise HTTPException(status_code=400, detail=str(e))

            # run inference; out file saved under results/.../vid_output/
            out_name = "audio_upload"

            try:
                avatar.inference(audio_path, out_name, fps, skip_save_images=False)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Inference error: {e}")

        out_mp4 = os.path.join(avatar.video_out_path, f"{out_name}.mp4")
        if not os.path.isfile(out_mp4):
            raise HTTPException(status_code=500, detail="Output MP4 not found after inference.")

    t1 = time.perf_counter()
    print(f"- infer total: {(t1 - t0):.2f}s | avatar={avatar_id} fps={fps} prep={preparation}")

    # return final video
    return FileResponse(out_mp4, media_type="video/mp4", filename=f"{avatar_id}.mp4")
