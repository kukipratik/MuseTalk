# api/server.py
import os, asyncio, torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from omegaconf import OmegaConf
from transformers import WhisperModel

from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing

LOCAL_MODELS = os.getenv("LOCAL_MODELS", "/opt/musetalk_models")

state = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "vae": None, "unet": None, "pe": None,
    "whisper": None, "audio_processor": None,
    "fp": None, "timesteps": None,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.backends.cudnn.benchmark = True
    # ---- preload models to CPU+VRAM
    vae, unet, pe = load_all_model(
        unet_model_path=f"{LOCAL_MODELS}/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config=f"{LOCAL_MODELS}/musetalkV15/musetalk.json",
        device=state["device"],
        vae_dir=f"{LOCAL_MODELS}/sd-vae",
    )
    pe = pe.half().to(state["device"])
    vae.vae = vae.vae.half().to(state["device"])
    unet.model = unet.model.half().to(state["device"])

    # Whisper & audio
    audio_processor = AudioProcessor(feature_extractor_path=f"{LOCAL_MODELS}/whisper")
    whisper = WhisperModel.from_pretrained(f"{LOCAL_MODELS}/whisper")
    whisper = whisper.to(device=state["device"], dtype=unet.model.dtype).eval()
    whisper.requires_grad_(False)

    # Face parser
    fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

    state.update(dict(
        vae=vae, unet=unet, pe=pe, audio_processor=audio_processor,
        whisper=whisper, fp=fp, timesteps=torch.tensor([0], device=state["device"])
    ))

    # Optional: tiny warm-up pass (jit kernels & first GPU memory touches)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        dummy_latents = torch.randn(1, 8, 32, 32, device=state["device"], dtype=torch.float16)
        _ = state["unet"].model(dummy_latents, state["timesteps"],
                                encoder_hidden_states=torch.randn(1, 1, 384, device=state["device"], dtype=torch.float16))
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    ok = all(state[k] is not None for k in ["vae","unet","pe","whisper","audio_processor","fp"])
    return {"ok": ok}

# Example inference endpoint (you’ll wire in your avatar paths)
# Keep a single GPU → no extra locking required with workers=1.
