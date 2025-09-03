import shutil
import uuid
from transformers import WhisperModel

import os, threading, cv2
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from musetalk.utils.utils import load_all_model, datagen, get_file_type, get_video_fps
from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder, get_bbox_range
from musetalk.utils.face_parsing import FaceParsing  # jaw/neck parsing & cheek protection
from musetalk.utils.blending import get_image
from musetalk.utils.audio_processor import AudioProcessor  # whisper features (50fps) + chunking

# ---------------------------- Data containers ----------------------------
@dataclass
class PreparedAvatar:
    avatar_id: str
    coord_list_cycle: List
    frame_list_cycle: List[np.ndarray]
    latent_list_cycle: List[torch.Tensor]
    bbox_shift_text: str
    parsing_mode: str
    extra_margin: int
    left_cheek_width: int
    right_cheek_width: int
    fps_default: int

# ---------------------------- Pipeline ----------------------------
class MuseTalkPipeline:
    """
    One-process pipeline for real-time lip sync:
      - UNet/VAE/PE on GPU (fp16)
      - Whisper on CPU
      - Avatar prepared once & cached
      - GPU inference under a lock for stable latency
    """
    def __init__(self,
                 device: Optional[torch.device] = None,
                 use_fp16: bool = True,
                 whisper_dir: str = "./models/whisper"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = torch.float16 if (use_fp16 and self.device.type == "cuda") else torch.float32

        self.vae = None
        self.unet = None
        self.pe = None
        self.timesteps = None
        self.whisper = None

        # audio feature extractor (Whisper-tiny on CPU) + helper
        self.audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
        # NOTE: AudioProcessor internally loads the HF Whisper model when needed.
        # We keep it on CPU to avoid GPU contention with UNet/VAE.

        # caches
        self.face_parser_cache: Dict[Tuple[int,int,int,str], FaceParsing] = {}
        self._avatars: Dict[str, PreparedAvatar] = {}

        # serialize GPU work
        self.gpu_lock = threading.Lock()

    # ------------------------ Model loading & warmup ------------------------
    def load_once(self,
                  unet_model_path: str = "./models/musetalkV15/unet.pth",
                  unet_config: str = "./models/musetalkV15/musetalk.json",
                  vae_type: str = "sd-vae"):
        # Load UNet / VAE / PosEnc
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type=vae_type,
            unet_config=unet_config,
            device=self.device
        )
        # fp16 cast if requested
        if self.weight_dtype == torch.float16:
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()
        # device & eval
        self.pe = self.pe.to(self.device).eval()
        self.vae.vae = self.vae.vae.to(self.device).eval()
        self.unet.model = self.unet.model.to(self.device).eval()

        self.timesteps = torch.tensor([0], device=self.device)  # MuseTalk runs at t=0

        # Load Whisper-tiny on CPU once
        src = self.audio_processor.feature_extractor_path \
              if os.path.isdir(self.audio_processor.feature_extractor_path) else "openai/whisper-tiny"
        self.whisper = WhisperModel.from_pretrained(src).to(torch.device("cpu")).eval()
        for p in self.whisper.parameters(): p.requires_grad = False

    def warmup(self, frames: int = 2):
        """
        Tiny synthetic forward to pre-initialize CUDA kernels/allocators.
        """
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.weight_dtype == torch.float16)):
            dummy_audio = torch.randn(frames, 50, 384, device=self.device, dtype=self.weight_dtype)
            af = self.pe(dummy_audio)
            dummy_latents = torch.randn(frames, 8, 32, 32, device=self.device, dtype=self.weight_dtype)
            _ = self.unet.model(dummy_latents, self.timesteps, encoder_hidden_states=af)

    # ------------------------ Avatar preparation (prep=True) ------------------------
    def prepare_avatar(self,
                       avatar_id: str,
                       video_path: str,
                       bbox_shift: int = 0,
                       fps: int = 25,
                       extra_margin: int = 10,
                       left_cheek_width: int = 90,
                       right_cheek_width: int = 90,
                       parsing_mode: str = "jaw") -> dict:
        """
        - Extract frames from the reference video (or image folder)
        - Detect bboxes/landmarks
        - VAE-encode crops -> latent list (masked+ref; 8ch)
        - Build cyclic buffers
        - Cache FaceParsing (jaw/neck) with cheek protection
        """
        # decide fps
        file_type = get_file_type(video_path)
        fps_detected = int(get_video_fps(video_path) or fps) if file_type == "video" else fps

        img_list = None
        tmp_frames_dir = None
        if file_type == "video":
            tmp_frames_dir = os.path.join("./results/prepared", f"frames_{uuid.uuid4().hex}")
            os.makedirs(tmp_frames_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            idx = 0
            ok, frame = cap.read()
            while ok:
                # write as numbered png to keep natural sort
                outp = os.path.join(tmp_frames_dir, f"{idx:06d}.png")
                cv2.imwrite(outp, frame)
                idx += 1
                ok, frame = cap.read()
            cap.release()
            img_list = [os.path.join(tmp_frames_dir, f) for f in sorted(os.listdir(tmp_frames_dir)) if f.endswith(".png")]
        else:
            # if user already provided an image folder, expand it to a list
            img_list = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        # feed a **list of image paths** to the repo util
        coord_list, frame_list = get_landmark_and_bbox(img_list, bbox_shift)

        if not coord_list or all(b == coord_placeholder for b in coord_list):
            # cleanup frames dir on failure
            if tmp_frames_dir is not None:
                try: shutil.rmtree(tmp_frames_dir)
                except: pass
            raise RuntimeError("No face detected during preparation; adjust bbox_shift or video quality")

        bbox_shift_text = get_bbox_range(frame_list, bbox_shift)  # accepts frames list

        # VAE latents per frame crop (256x256)
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            y2m = min(y2 + extra_margin, frame.shape[0])
            crop = frame[y1:y2m, x1:x2]
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(crop)  # [1, 8, 32, 32]
            input_latent_list.append(latents)

        # cyclic buffers to smooth loop boundaries
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        latent_list_cycle = input_latent_list + input_latent_list[::-1]

        # cache face parser per shape/mode
        fp_key = (left_cheek_width, right_cheek_width, extra_margin, parsing_mode)
        if fp_key not in self.face_parser_cache:
            self.face_parser_cache[fp_key] = FaceParsing(
                left_cheek_width=left_cheek_width,
                right_cheek_width=right_cheek_width
            )

        self._avatars[avatar_id] = PreparedAvatar(
            avatar_id=avatar_id,
            coord_list_cycle=coord_list_cycle,
            frame_list_cycle=frame_list_cycle,
            latent_list_cycle=latent_list_cycle,
            bbox_shift_text=bbox_shift_text,
            parsing_mode=parsing_mode,
            extra_margin=extra_margin,
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
            fps_default=fps_detected
        )

        if tmp_frames_dir is not None:
            try: shutil.rmtree(tmp_frames_dir)
            except: pass

        return {
            "avatar_id": avatar_id,
            "fps": fps_detected,
            "num_frames": len(frame_list),
            "bbox_shift_text": bbox_shift_text
        }

    # ------------------------ Lip sync (prep=False) ------------------------
    def lip_sync(self,
                 avatar_id: str,
                 audio_path: str,
                 out_dir: str,
                 output_basename: Optional[str] = None,
                 fps: int = 25,
                 batch_size: int = 8,
                 audio_padding_left: int = 2,
                 audio_padding_right: int = 2) -> str:
        """
        CPU Whisper -> UNet+VAE (GPU) -> Blend -> mp4 (mux audio)
        Returns final mp4 path.
        """
        if avatar_id not in self._avatars:
            raise KeyError(f"Avatar '{avatar_id}' is not prepared. Call /prepare_avatar first.")

        pav = self._avatars[avatar_id]
        os.makedirs(out_dir, exist_ok=True)
        out_name = output_basename or f"{avatar_id}_{os.path.splitext(os.path.basename(audio_path))[0]}"
        tmp_video = os.path.join(out_dir, f"{out_name}_noaudio.mp4")
        final_video = os.path.join(out_dir, f"{out_name}.mp4")

        # ---- 1) Audio -> Whisper features (CPU) ----
        #   AudioProcessor builds whisper input features and slices them to 25fps chunks
        whisper_input_features, librosa_len = self.audio_processor.get_audio_feature(audio_path)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            device=torch.device("cpu"),
            weight_dtype=torch.float32,
            whisper=self.whisper,               # AudioProcessor manages its own whisper model
            librosa_length=librosa_len,
            fps=fps,
            audio_padding_length_left=audio_padding_left,
            audio_padding_length_right=audio_padding_right
        )

        # ---- 2) GPU inference loop (UNet + VAE) ----
        # face parser from cache (jaw/neck with cheek protection)
        fp_key = (pav.left_cheek_width, pav.right_cheek_width, pav.extra_margin, pav.parsing_mode)
        face_parser = self.face_parser_cache[fp_key]

        # use OpenCV to write a silent mp4 quickly
        H, W = pav.frame_list_cycle[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # h264
        writer = cv2.VideoWriter(tmp_video, fourcc, fps, (W, H))

        try:
            with self.gpu_lock:
                gen = datagen(
                    whisper_chunks=whisper_chunks,
                    vae_encode_latents=pav.latent_list_cycle,
                    batch_size=batch_size,
                    delay_frame=0,
                    device="cuda:0" if self.device.type == "cuda" else "cpu"
                )

                for i, (whisper_batch, latent_batch) in enumerate(gen):
                    whisper_batch = whisper_batch.to(self.device, dtype=self.weight_dtype, non_blocking=True)
                    latent_batch = latent_batch.to(self.device, dtype=self.weight_dtype, non_blocking=True)

                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.weight_dtype == torch.float16)):
                        audio_feature_batch = self.pe(whisper_batch)  # [B,50,384] -> PE features
                        pred_latents = self.unet.model(
                            latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch
                        ).sample
                        recon = self.vae.decode_latents(pred_latents)  # numpy RGB uint8

                    # Blend reconstructed mouth into original frames, write to video
                    for j, res_frame in enumerate(recon):
                        idx = (i * batch_size + j) % len(pav.coord_list_cycle)
                        bbox = pav.coord_list_cycle[idx]
                        ori = pav.frame_list_cycle[idx].copy()
                        x1, y1, x2, y2 = bbox
                        y2m = min(y2 + pav.extra_margin, ori.shape[0])
                        # resize predicted 256x256 to bbox region
                        res_rsz = cv2.resize(res_frame.astype(np.uint8), (max(1, x2 - x1), max(1, y2m - y1)))
                        combined = get_image(ori, res_rsz, [x1, y1, x2, y2],
                                             mode=pav.parsing_mode, fp=face_parser)
                        writer.write(combined[:, :, ::-1])  # RGB->BGR
        finally:
            writer.release()

        # ---- 3) Mux audio (copy video stream) ----
        # Requires ffmpeg system binary installed
        os.system(
            f'ffmpeg -y -loglevel error -i "{tmp_video}" -i "{audio_path}" '
            f'-c:v copy -c:a aac -shortest "{final_video}"'
        )
        try: os.remove(tmp_video)
        except: pass

        return final_video

    # ------------------------ helpers ------------------------
    def has_avatar(self, avatar_id: str) -> bool:
        return avatar_id in self._avatars

    def list_avatars(self) -> List[str]:
        return list(self._avatars.keys())
