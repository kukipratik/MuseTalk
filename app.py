# server_fastapi.py
import os, uuid, time, logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from server.muse_pipeline import MuseTalkPipeline

# ----------------- logging -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("musetalk-api")

# ----------------- app & pipeline -----------------
app = FastAPI(title="MuseTalk FastAPI", version="1.1.0")
PIPE = MuseTalkPipeline(use_fp16=True)  # GPU fp16 for UNet/VAE; Whisper on CPU

OUT_DIR_PREP = "./results/prepared"
OUT_DIR_VID = "./results/output"
os.makedirs(OUT_DIR_PREP, exist_ok=True)
os.makedirs(OUT_DIR_VID, exist_ok=True)

# ----------------- startup: load & warmup -----------------
@app.on_event("startup")
def load_models():
    t0 = time.perf_counter()
    PIPE.load_once(
        unet_model_path="./models/musetalkV15/unet.pth",
        unet_config="./models/musetalkV15/musetalk.json",
        vae_type="sd-vae"
    )
    PIPE.warmup(frames=2)
    t1 = time.perf_counter()
    log.info(f"[startup] models loaded+warm in {(t1 - t0):.2f}s | device={PIPE.device} | dtype={PIPE.weight_dtype}")

# ----------------- endpoints -----------------
@app.post("/prepare_avatar")
async def prepare_avatar(
    avatar_id: str = Form(..., description="ID to cache the prepared assets (e.g., lisa/lucy/mark)"),
    video: UploadFile = File(..., description="Reference video (preferred) or zipped images"),
    bbox_shift: int = Form(0),
    fps: int = Form(25),
    extra_margin: int = Form(10),
    left_cheek_width: int = Form(90),
    right_cheek_width: int = Form(90),
    parsing_mode: str = Form("jaw")
):
    # Save the uploaded ref video
    ref_path = os.path.join(OUT_DIR_PREP, f"{uuid.uuid4()}_{video.filename}")
    with open(ref_path, "wb") as f:
        f.write(await video.read())

    t0 = time.perf_counter()
    log.info(
        f"[/prepare_avatar] hit | avatar_id={avatar_id} video={video.filename} "
        f"bbox_shift={bbox_shift} fps={fps} parsing={parsing_mode} "
        f"cheeks=({left_cheek_width},{right_cheek_width}) extra_margin={extra_margin}"
    )

    try:
        info = PIPE.prepare_avatar(
            avatar_id=avatar_id,
            video_path=ref_path,
            bbox_shift=bbox_shift,
            fps=fps,
            extra_margin=extra_margin,
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
            parsing_mode=parsing_mode
        )
    except Exception as e:
        log.exception("[/prepare_avatar] failed")
        raise HTTPException(status_code=400, detail=str(e))

    took = round(time.perf_counter() - t0, 2)
    log.info(f"[/prepare_avatar] done in {took}s | avatar_id={avatar_id} frames={info.get('num_frames')}")

    return JSONResponse({"ok": True, "avatar_id": avatar_id, "prep_info": info, "took_sec": took})


@app.post("/lip_sync")
async def lip_sync(
    avatar_id: str = Form(...),
    audio: UploadFile = File(..., description="Driving audio (wav/mp3)"),
    fps: int = Form(25),
    batch_size: int = Form(8),
    audio_padding_left: int = Form(2),
    audio_padding_right: int = Form(2)
):
    if not PIPE.has_avatar(avatar_id):
        raise HTTPException(status_code=404, detail=f"Avatar '{avatar_id}' not prepared")

    # Save uploaded audio
    audio_path = os.path.join(OUT_DIR_VID, f"{uuid.uuid4()}_{audio.filename}")
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    t0 = time.perf_counter()
    log.info(
        f"[/lip_sync] hit | avatar_id={avatar_id} audio={audio.filename} "
        f"fps={fps} batch_size={batch_size} padL/R=({audio_padding_left},{audio_padding_right})"
    )

    try:
        out_video = PIPE.lip_sync(
            avatar_id=avatar_id,
            audio_path=audio_path,
            out_dir=OUT_DIR_VID,
            output_basename=f"{avatar_id}_{os.path.splitext(audio.filename)[0]}",
            fps=fps,
            batch_size=batch_size,
            audio_padding_left=audio_padding_left,
            audio_padding_right=audio_padding_right
        )
    except Exception as e:
        log.exception("[/lip_sync] failed")
        raise HTTPException(status_code=400, detail=str(e))

    took = round(time.perf_counter() - t0, 2)
    log.info(f"[/lip_sync] done in {took}s | avatar_id={avatar_id} -> {os.path.basename(out_video)}")

    return FileResponse(
        out_video,
        media_type="video/mp4",
        filename=os.path.basename(out_video),
        headers={"X-Processing-Time-Sec": str(took)}
    )
