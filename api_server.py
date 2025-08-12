from fastapi import FastAPI
import torch
from musetalk.utils.utils import load_all_model

app = FastAPI()

# Load models at API startup (keeps them in memory)
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
def infer(audio_path: str):
    # Replace with your actual inference function
    return {"status": "success", "audio_path": audio_path}
