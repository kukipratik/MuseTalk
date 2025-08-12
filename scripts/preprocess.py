import os
import sys
import argparse
import subprocess
from typing import Tuple, List, Union

import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import json
import cv2
import decord


def fast_check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def ensure_ffmpeg_on_path(ffmpeg_path: str) -> None:
    """Put ffmpeg on PATH if it's not already available."""
    if fast_check_ffmpeg():
        return
    print("Adding ffmpeg to PATH")
    path_separator = ';' if sys.platform == 'win32' else ':'
    os.environ["PATH"] = f"{ffmpeg_path}{path_separator}{os.environ.get('PATH','')}"
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")


# --------- Heavy deps are imported lazily inside classes/functions ----------

class AnalyzeFace:
    def __init__(self, device: Union[str, torch.device], config_file: str, checkpoint_file: str):
        """
        Initialize face detector + keypoint model.
        Heavy imports happen here to avoid slowing down unrelated imports.
        """
        from musetalk.utils.face_detection import FaceAlignment, LandmarksType  # lazy
        from mmpose.apis import init_model as mmpose_init_model  # lazy

        self.device = device
        self.FaceAlignment = FaceAlignment
        self.LandmarksType = LandmarksType

        # init models
        self.dwpose = mmpose_init_model(config_file, checkpoint_file, device=self.device)
        self.facedet = FaceAlignment(LandmarksType._2D, flip_input=False, device=self.device)

        # keep light mmpose helpers locally to avoid global imports
        from mmpose.apis import inference_topdown as _inference_topdown
        from mmpose.structures import merge_data_samples as _merge_data_samples
        self._inference_topdown = _inference_topdown
        self._merge_data_samples = _merge_data_samples

    def __call__(self, im: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Detect faces and keypoints in the given image.
        Returns (face_landmarks, face_bboxes)
        """
        try:
            if im.ndim == 3:
                im = np.expand_dims(im, axis=0)
            elif im.ndim != 4 or im.shape[0] != 1:
                raise ValueError("Input image must have shape (1, H, W, C)")

            bbox = self.facedet.get_detections_for_batch(np.asarray(im))
            results = self._inference_topdown(self.dwpose, np.asarray(im)[0])
            results = self._merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_land_mark = keypoints[0][23:91].astype(np.int32)

            return face_land_mark, bbox

        except Exception as e:
            print(f"Error during face analysis: {e}")
            return np.array([]), []


def convert_video(org_path: str, dst_path: str, vid_list: List[str]) -> None:
    """Convert videos to 25 fps libx264 yuv420p for consistency."""
    os.makedirs(dst_path, exist_ok=True)
    for idx, vid in enumerate(vid_list):
        if not vid.endswith('.mp4'):
            continue
        org_vid_path = os.path.join(org_path, vid)
        dst_vid_path = os.path.join(dst_path, vid)

        if org_vid_path != dst_vid_path:
            cmd = [
                "ffmpeg", "-hide_banner", "-y", "-i", org_vid_path,
                "-r", "25", "-crf", "15", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", dst_vid_path
            ]
            subprocess.run(cmd, check=True)

        if idx % 1000 == 0:
            print(f"### {idx} videos converted ###")


def segment_video(org_path: str, dst_path: str, vid_list: List[str], segment_duration: int = 30) -> None:
    """Split long videos into fixed-length segments (copy/mux only)."""
    os.makedirs(dst_path, exist_ok=True)
    for vid in vid_list:
        if not vid.endswith('.mp4'):
            continue
        input_file = os.path.join(org_path, vid)
        original_filename = os.path.basename(input_file)
        cmd = [
            'ffmpeg', '-i', input_file, '-c', 'copy', '-map', '0',
            '-segment_time', str(segment_duration), '-f', 'segment',
            '-reset_timestamps', '1',
            os.path.join(dst_path, f'clip%03d_{original_filename}')
        ]
        subprocess.run(cmd, check=True)


def extract_audio(org_path: str, dst_path: str, vid_list: List[str]) -> None:
    """Extract mono 16k PCM WAV from each MP4."""
    os.makedirs(dst_path, exist_ok=True)
    for vid in vid_list:
        if not vid.endswith('.mp4'):
            continue
        video_path = os.path.join(org_path, vid)
        audio_output_path = os.path.join(dst_path, os.path.splitext(vid)[0] + ".wav")
        try:
            cmd = [
                'ffmpeg', '-hide_banner', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le', '-f', 'wav',
                '-ar', '16000', '-ac', '1', audio_output_path,
            ]
            subprocess.run(cmd, check=True)
            # print(f"Audio saved to: {audio_output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio from {vid}: {e}")


def split_data(video_files: List[str], val_list_hdtf: List[str]) -> (List[str], List[str]):
    """Split into train/val sets based on provided validation ids."""
    val_files = [f for f in video_files if any(val_id in f for val_id in val_list_hdtf)]
    train_files = [f for f in video_files if f not in val_files]
    return train_files, val_files


def save_list_to_file(file_path: str, data_list: List[str]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(f"{item}\n")


def generate_train_list(cfg):
    train_file_path = cfg.video_clip_file_list_train
    val_file_path = cfg.video_clip_file_list_val
    val_list_hdtf = cfg.val_list_hdtf

    meta_list = os.listdir(cfg.meta_root)
    sorted_meta_list = sorted(meta_list)  # noqa: F841

    train_files, val_files = split_data(meta_list, val_list_hdtf)
    save_list_to_file(train_file_path, train_files)
    save_list_to_file(val_file_path, val_files)
    print(val_list_hdtf)


def analyze_video(org_path: str, dst_path: str, vid_list: List[str]) -> None:
    """Run face/bbox/landmark analysis and dump JSON meta per clip."""
    os.makedirs(dst_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
    checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'

    analyze_face = AnalyzeFace(device, config_file, checkpoint_file)

    for vid in tqdm(vid_list, desc="Processing videos"):
        if not vid.endswith('.mp4'):
            continue
        vid_path = os.path.join(org_path, vid)
        wav_path = vid_path.replace(".mp4", ".wav")
        vid_meta = os.path.join(dst_path, os.path.splitext(vid)[0] + ".json")
        if os.path.exists(vid_meta):
            continue

        total_bbox_list, total_pts_list = [], []
        isvalid = True

        try:
            cap = decord.VideoReader(vid_path, fault_tol=1)
        except Exception as e:
            print(e)
            continue

        total_frames = len(cap)
        face_height = face_width = 0  # default values
        for frame_idx in range(total_frames):
            frame = cap[frame_idx]
            if frame_idx == 0:
                video_height, video_width, _ = frame.shape
            frame_bgr = cv2.cvtColor(frame.asnumpy(), cv2.COLOR_BGR2RGB)

            pts_list, bbox_list = analyze_face(frame_bgr)

            if len(bbox_list) > 0 and None not in bbox_list:
                bbox = bbox_list[0]
            else:
                isvalid = False
                bbox = []
                print(f"set isvalid to False as broken img in {frame_idx} of {vid}")
                break

            if len(pts_list) > 0 and pts_list is not None:
                pts = pts_list.tolist()
            else:
                isvalid = False
                pts = []
                break

            if frame_idx == 0:
                x1, y1, x2, y2 = bbox
                face_height, face_width = y2 - y1, x2 - x1

            total_pts_list.append(pts)
            total_bbox_list.append(bbox)

        meta_data = {
            "mp4_path": vid_path,
            "wav_path": wav_path,
            "video_size": [video_height, video_width],
            "face_size": [face_height, face_width],
            "frames": total_frames,
            "face_list": total_bbox_list,
            "landmark_list": total_pts_list,
            "isvalid": isvalid,
        }
        with open(vid_meta, 'w') as f:
            json.dump(meta_data, f, indent=4)


def main(cfg, ffmpeg_path: str):
    ensure_ffmpeg_on_path(ffmpeg_path)

    # Ensure required directories
    os.makedirs(cfg.video_root_25fps, exist_ok=True)
    os.makedirs(cfg.video_audio_clip_root, exist_ok=True)
    os.makedirs(cfg.meta_root, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_file_list), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_clip_file_list_train), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_clip_file_list_val), exist_ok=True)

    vid_list = sorted(os.listdir(cfg.video_root_raw))

    # 0. Save original video file list
    with open(cfg.video_file_list, 'w') as file:
        for vid in vid_list:
            file.write(vid + '\n')

    # 1. Convert videos to 25 FPS
    convert_video(cfg.video_root_raw, cfg.video_root_25fps, vid_list)

    # 2. Segment videos into clips
    segment_video(cfg.video_root_25fps, cfg.video_audio_clip_root, vid_list, segment_duration=cfg.clip_len_second)

    # 3. Extract audio
    clip_vid_list = sorted(os.listdir(cfg.video_audio_clip_root))
    extract_audio(cfg.video_audio_clip_root, cfg.video_audio_clip_root, clip_vid_list)

    # 4. Generate video metadata
    analyze_video(cfg.video_audio_clip_root, cfg.meta_root, clip_vid_list)

    # 5. Generate training and validation lists
    generate_train_list(cfg)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/preprocess.yaml")
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(config, ffmpeg_path=args.ffmpeg_path)
