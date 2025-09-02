import os
import warnings
warnings.filterwarnings("ignore", message="Decorating classes is deprecated")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

import argparse
from types import SimpleNamespace
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
from torch.cuda.amp import autocast
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
from transformers import WhisperModel

# light utils
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen, load_all_model
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.audio_processor import AudioProcessor

import shutil
import threading
import queue
import time
import subprocess


# ---------------- Runtime injection ----------------
R = SimpleNamespace(
    device=None, vae=None, unet=None, pe=None,
    whisper=None, audio_processor=None, timesteps=None,
    fp=None, weight_dtype=None
)

def inject_runtime(*, device, vae, unet, pe, whisper, audio_processor, timesteps, fp, weight_dtype):
    """Called by FastAPI at startup (or by CLI main) to register hot models once."""
    R.device = device
    R.vae = vae
    R.unet = unet
    R.pe = pe
    R.whisper = whisper
    R.audio_processor = audio_processor
    R.timesteps = timesteps
    R.fp = fp
    R.weight_dtype = weight_dtype

    # small perf nudge: channels-last on conv-heavy modules
    try:
        R.unet.model.to(memory_format=torch.channels_last)
        R.vae.vae.to(memory_format=torch.channels_last)
        R.pe.to(memory_format=torch.channels_last)
    except Exception:
        pass

def _assert_runtime():
    missing = [k for k in ("device","vae","unet","pe","whisper","audio_processor","timesteps","fp","weight_dtype")
               if getattr(R, k) is None]
    if missing:
        raise RuntimeError(f"Runtime not injected; missing: {missing}. "
                           "If using FastAPI, call inject_runtime() at app startup. "
                           "If using CLI, run this module directly so __main__ injects it.")


# ---------------- Helpers ----------------
def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10_000_000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path, exist_ok=True)


def _read_imgs_fast(file_list):
    """Super-light image reader that avoids importing heavy preprocessing (DWPose)."""
    imgs = []
    for p in file_list:
        im = cv2.imread(p)
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imgs.append(im)
    return imgs


# ---------------- Core Avatar runner ----------------
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        _assert_runtime()

        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        if args.version == "v15":
            self.base_path = f"./results/{args.version}/avatars/{avatar_id}"
        else:
            self.base_path = f"./results/avatars/{avatar_id}"

        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": args.version
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        print(f"[Avatar] init avatar_id={avatar_id} bbox_shift={bbox_shift} prep={preparation}")
        self.init()

    def init(self):
        # Server-safe behavior (no interactive prompts):
        # - If preparation=True → (re)create deterministically
        # - If preparation=False → load cache or error out clearly
        if self.preparation:
            if os.path.exists(self.avatar_path):
                shutil.rmtree(self.avatar_path)
            osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
            self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist. Run with preparation=True once for this avatar.")
                sys.exit(1)

            # if bbox_shift changed, auto-recreate
            try:
                with open(self.avatar_info_path, "r") as f:
                    avatar_info = json.load(f)
            except Exception:
                avatar_info = {}

            if avatar_info.get('bbox_shift') != self.avatar_info['bbox_shift']:
                shutil.rmtree(self.avatar_path)
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
            else:
                self._load_cached_materials()

    def _load_cached_materials(self):
        """Fast path: no heavy imports. Avoids DWPose/S3FD on prep=False startup."""
        # safer CPU map; move to GPU only when needed
        self.input_latent_list_cycle = torch.load(self.latents_out_path, map_location="cpu")
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)

        input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = _read_imgs_fast(input_img_list)

        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = _read_imgs_fast(input_mask_list)

    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"copy files in {self.video_path}")
            files = [fn for fn in sorted(os.listdir(self.video_path)) if fn.lower().endswith(".png")]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")

        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        print("extracting landmarks...")
        # heavy DWPose/S3FD import delayed to here:
        from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs  # noqa: F401

        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx += 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if args.version == "v15":
                y2 = min(y2 + args.extra_margin, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = R.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            x1, y1, x2, y2 = self.coord_list_cycle[i]
            mode = args.parsing_mode if args.version == "v15" else "raw"
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=R.fp, mode=mode)

            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle.append(crop_box)
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)
        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

    def process_frames(self, res_frame_queue, video_len, skip_save_images):
        print(video_len)
        while True:
            if self.idx >= video_len - 1:
                break
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[self.idx % len(self.coord_list_cycle)]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % len(self.frame_list_cycle)])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception:
                continue
            mask = self.mask_list_cycle[self.idx % len(self.mask_list_cycle)]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % len(self.mask_coords_list_cycle)]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            if not skip_save_images:
                bgr = cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.idx += 1
            
    def _ensure_rgb(self, img: np.ndarray) -> np.ndarray:
        """
        Best-effort guard: if blue channel is suspiciously dominant on a skin-heavy crop,
        assume it's BGR and swap to RGB. Otherwise return as-is.
        """
        if img.ndim != 3 or img.shape[2] != 3:
            return img
        # sample center 48x48 patch to avoid edges
        h, w, _ = img.shape
        y0, y1 = h//2 - 24, h//2 + 24
        x0, x1 = w//2 - 24, w//2 + 24
        patch = img[max(0,y0):min(h,y1), max(0,x0):min(w,x1)]
        b, g, r = patch[...,0].mean(), patch[...,1].mean(), patch[...,2].mean()
        # if blue >> red, it's almost surely BGR for skin; swap
        return img[..., ::-1] if (b > r + 12 and b > g + 6) else img

    @torch.no_grad()
    def inference(self, audio_path, out_vid_name, fps, skip_save_images):
        _assert_runtime()

        os.makedirs(self.avatar_path + '/tmp', exist_ok=True)
        print("start inference")
        # -------- audio feature extraction --------
        t0 = time.time()
        whisper_input_features, librosa_length = R.audio_processor.get_audio_feature(
            audio_path, weight_dtype=R.weight_dtype
        )
        whisper_chunks = R.audio_processor.get_whisper_chunk(
            whisper_input_features,
            R.device,
            R.weight_dtype,
            R.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
        print(f"processing audio:{audio_path} costs {(time.time() - t0) * 1000:.2f}ms")

        # -------- UNet+VAE loop --------
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0

        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num, skip_save_images))
        process_thread.start()

        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        t1 = time.time()

        with torch.inference_mode():
            # autocast for UNet/VAE forward (weights already half)
            for _, (whisper_batch, latent_batch) in enumerate(
                tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))
            ):
                with autocast(dtype=torch.float16):
                    audio_feature_batch = R.pe(whisper_batch.to(R.device, non_blocking=True))
                    latent_batch = latent_batch.to(device=R.device, dtype=R.unet.model.dtype, non_blocking=True)
                    latent_batch = latent_batch.contiguous(memory_format=torch.channels_last)

                    pred_latents = R.unet.model(
                        latent_batch, R.timesteps, encoder_hidden_states=audio_feature_batch
                    ).sample

                    # decode in fp16 under autocast
                    recon = R.vae.decode_latents(pred_latents)

                for res_frame in recon:
                    if res_frame.dtype != np.uint8:
                        res_frame = res_frame.astype(np.uint8)

                    # **Remove** unconditional cvtColor. Use the guard instead:
                    res_frame = self._ensure_rgb(res_frame)

                    # Keep everything in RGB for blending (your frames/masks are RGB)
                    res_frame_queue.put(res_frame)

        process_thread.join()

        if args.skip_save_images:
            print(f'Total process time of {video_num} frames without saving images = {time.time() - t1:.3f}s')
        else:
            print(f'Total process time of {video_num} frames including saving images = {time.time() - t1:.3f}s')

        if out_vid_name is not None and not args.skip_save_images:
            # faster mux + explicit color tags
            cmd_img2video = (
                f"ffmpeg -y -v warning -r {fps} -f image2 "
                f"-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 "
                f"-i {self.avatar_path}/tmp/%08d.jpg "
                f"-c:v libx264 -preset ultrafast -tune zerolatency "
                f"-pix_fmt yuv420p -color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 "
                f"{self.avatar_path}/temp.mp4"
            )
            print(cmd_img2video); os.system(cmd_img2video)

            output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")
            cmd_combine_audio = (
                f"ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 "
                f"-c:v copy -c:a aac -shortest {output_vid}"
            )
            print(cmd_combine_audio); os.system(cmd_combine_audio)

            os.remove(f"{self.avatar_path}/temp.mp4")
            shutil.rmtree(f"{self.avatar_path}/tmp", ignore_errors=True)
            print(f"result is save to {output_vid}")
        print("\n")


# ---------------- CLI entry (still supported) ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Version of MuseTalk: v1 or v15")
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--vae_dir", type=str, default="./models/sd-vae", help="Directory of local SD VAE weights")
    parser.add_argument("--unet_config", type=str, default="./models/musetalkV15/musetalk.json", help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth", help="Path to UNet model weights")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--inference_config", type=str, default="configs/inference/realtime.yaml")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    parser.add_argument("--result_dir", default='./results', help="Directory for output results")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for inference")
    parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    parser.add_argument("--use_saved_coord", action="store_true", help='Use saved coordinates to save time')
    parser.add_argument("--saved_coord", action="store_true", help='Save coordinates for future use')
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--skip_save_images", action="store_true", help="Whether skip saving images for better generation speed calculation")
    args = parser.parse_args()
    default_bbox_shift = args.bbox_shift

    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ.get('PATH','')}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    
    # Debug log to confirm precision settings
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, TF32={torch.backends.cuda.matmul.allow_tf32}, "
        f"matmul_precision={torch.get_float32_matmul_precision()}")

    # Load models once
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device,
        vae_dir=args.vae_dir,
    )
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    # channels-last (small boost on Ampere+)
    try:
        unet.model.to(memory_format=torch.channels_last)
        vae.vae.to(memory_format=torch.channels_last)
        pe.to(memory_format=torch.channels_last)
    except Exception:
        pass

    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=unet.model.dtype).eval()
    whisper.requires_grad_(False)

    if args.version == "v15":
        fp = FaceParsing(left_cheek_width=args.left_cheek_width, right_cheek_width=args.right_cheek_width)
    else:
        fp = FaceParsing()

    timesteps = torch.tensor([0], device=device)

    # Inject for this process (CLI mode)
    inject_runtime(
        device=device, vae=vae, unet=unet, pe=pe, whisper=whisper,
        audio_processor=audio_processor, timesteps=timesteps,
        fp=fp, weight_dtype=unet.model.dtype
    )

    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)

    for avatar_id in inference_config:
        data_preparation = inference_config[avatar_id]["preparation"]
        video_path = inference_config[avatar_id]["video_path"]
        cfg_shift = inference_config[avatar_id].get("bbox_shift", default_bbox_shift)
        print(f"[Avatar:{avatar_id}] Using bbox_shift={cfg_shift}")
        avatar = Avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=cfg_shift,
            batch_size=args.batch_size,
            preparation=data_preparation
        )
        audio_clips = inference_config[avatar_id]["audio_clips"]
        for audio_num, audio_path in audio_clips.items():
            print("Inferring using:", audio_path)
            avatar.inference(audio_path, audio_num, args.fps, args.skip_save_images)
