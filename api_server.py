# inside api/server.py
from fastapi import FastAPI
import torch
from musetalk.utils.utils import load_all_model
import subprocess

app = FastAPI()

# Load once (same as warm-up script)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vae, unet, pe = load_all_model(
    unet_model_path="models/musetalkV15/unet.pth",
    vae_type="sd-vae",
    unet_config="models/musetalkV15/musetalk.json",
    device=device,
    vae_dir="models/sd-vae",
)

@app.get("/health")
def health_check():
    return {"status": "ok", "msg": "Server is warm & ready!"}

@app.post("/infer")
def infer(audio_path: str, prep: bool = False):
    """
    Run realtime_inference with the already-loaded environment.
    Assumes audio_path is relative to /root/musetalk_hot or /workspace/MuseTalk.
    """
    cmd = [
        "python", "-m", "scripts.realtime_inference",
        "--inference_config", "configs/inference/realtime.yaml",
        "--unet_model_path", "models/musetalkV15/unet.pth",
        "--unet_config", "models/musetalkV15/musetalk.json",
        "--version", "v15",
        "--vae_type", "sd-vae",
        "--vae_dir", "models/sd-vae",
        "--fps", "25"
    ]
    if not prep:
        # If no preparation needed, adjust realtime.yaml or params accordingly
        pass

    subprocess.run(cmd, check=True)
    return {"status": "success", "audio_path": audio_path}
