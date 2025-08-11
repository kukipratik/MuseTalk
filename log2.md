//1st try
preparation: True: 25 fps, 720 p, 1sec avatar video:: audio: 7 sec -> took: 2min 8 sec

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# sh inference.sh v1.5 realtime
2025-08-11 06:49:09.573440: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-11 06:49:10.223477: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-11 06:49:14.277783: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loads checkpoint by local backend from path: ./models/dwpose/dw-ll_ucoco_384.pth
cuda start
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/utils/_contextlib.py:125: UserWarning: Decorating classes is deprecated and will be disabled in future versions. You should only decorate functions or methods. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.
  warnings.warn("Decorating classes is deprecated and will be disabled in "
An error occurred while trying to fetch models/sd-vae: Error no file named diffusion_pytorch_model.safetensors found in directory models/sd-vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
load unet model from ./models/musetalkV15/unet.pth
{'lisa1': {'preparation': True, 'bbox_shift': 5, 'video_path': 'data/video/lisa1.mp4', 'audio_clips': {'audio_1': 'data/audio/1.mp3'}}}
*********************************
  creating avator: lisa1
*********************************
preparing data materials ... ...
extracting landmarks...
reading images...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 28/28 [00:00<00:00, 78.84it/s]
get key_landmark and face bounding boxes with the default value
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 28/28 [00:06<00:00,  4.28it/s]
********************************************bbox_shift parameter adjustment**********************************************************
Total frame:「28」 Manually adjust range : [ -12~10 ] , the current value: 0
*************************************************************************************************************************************
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56/56 [00:05<00:00,  9.63it/s]
Inferring using: data/audio/1.mp3
start inference
processing audio:data/audio/1.mp3 costs 6576.58314704895ms
178
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:08<00:00,  1.01it/s]
Total process time of 178 frames including saving images = 11.99127721786499s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa1/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa1/temp.mp4
ffmpeg -y -v warning -i data/audio/1.mp3 -i ./results/v15/avatars/lisa1/temp.mp4 ./results/v15/avatars/lisa1/vid_output/audio_1.mp4
[mp3 @ 0x55db262ba7c0] Estimating duration from bitrate, this may be inaccurate
result is save to ./results/v15/avatars/lisa1/vid_output/audio_1.mp4

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk#

//2nd try
preparation: True: 25 fps, 720 p, 1sec avatar video:: audio: 8 sec -> took: 2min 8 sec