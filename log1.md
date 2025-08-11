//1st try
preparation: True: 25 fps, 720 p, 2.6sec avatar video:: audio: 7 sec -> took: 1 min 58 sec

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# sh inference.sh v1.5 realtime
2025-08-11 06:20:59.133306: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-11 06:20:59.187236: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-11 06:21:02.926794: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loads checkpoint by local backend from path: ./models/dwpose/dw-ll_ucoco_384.pth
cuda start
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/utils/_contextlib.py:125: UserWarning: Decorating classes is deprecated and will be disabled in future versions. You should only decorate functions or methods. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.
  warnings.warn("Decorating classes is deprecated and will be disabled in "
An error occurred while trying to fetch models/sd-vae: Error no file named diffusion_pytorch_model.safetensors found in directory models/sd-vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
load unet model from ./models/musetalkV15/unet.pth
{'lisa': {'preparation': True, 'bbox_shift': 5, 'video_path': 'data/video/lisa26.mp4', 'audio_clips': {'audio_3': 'data/audio/1.mp3'}}}
lisa exists, Do you want to re-create it ? (y/n)y
*********************************
  creating avator: lisa
*********************************
preparing data materials ... ...
extracting landmarks...
reading images...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 67/67 [00:00<00:00, 105.40it/s]
get key_landmark and face bounding boxes with the default value
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 67/67 [00:10<00:00,  6.46it/s]
********************************************bbox_shift parameter adjustment**********************************************************
Total frame:「67」 Manually adjust range : [ -11~11 ] , the current value: 0
*************************************************************************************************************************************
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:12<00:00, 10.68it/s]
Inferring using: data/audio/1.mp3
start inference
processing audio:data/audio/1.mp3 costs 3358.74342918396ms
178
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:08<00:00,  1.05it/s]
Total process time of 178 frames including saving images = 10.780780553817749s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa/temp.mp4
ffmpeg -y -v warning -i data/audio/1.mp3 -i ./results/v15/avatars/lisa/temp.mp4 ./results/v15/avatars/lisa/vid_output/audio_3.mp4
[mp3 @ 0x5652b37227c0] Estimating duration from bitrate, this may be inaccurate
result is save to ./results/v15/avatars/lisa/vid_output/audio_3.mp4
(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# 

// 2nd try
preparation: False: 25 fps, 720 p, 2.6sec avatar video:: audio: 8 sec -> took: 1min 10 sec

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# sh inference.sh v1.5 realtime
2025-08-11 06:26:13.345326: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-11 06:26:13.399599: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-11 06:26:16.680974: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loads checkpoint by local backend from path: ./models/dwpose/dw-ll_ucoco_384.pth
cuda start
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/utils/_contextlib.py:125: UserWarning: Decorating classes is deprecated and will be disabled in future versions. You should only decorate functions or methods. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.
  warnings.warn("Decorating classes is deprecated and will be disabled in "
An error occurred while trying to fetch models/sd-vae: Error no file named diffusion_pytorch_model.safetensors found in directory models/sd-vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
load unet model from ./models/musetalkV15/unet.pth
{'lisa': {'preparation': False, 'bbox_shift': 5, 'video_path': 'data/video/lisa26.mp4', 'audio_clips': {'audio_3': 'data/audio/2.mp3'}}}
reading images...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:01<00:00, 92.56it/s]
reading images...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:00<00:00, 373.42it/s]
Inferring using: data/audio/2.mp3
start inference
processing audio:data/audio/2.mp3 costs 3344.935655593872ms
204
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:07<00:00,  1.42it/s]
Total process time of 204 frames including saving images = 11.866919994354248s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa/temp.mp4
ffmpeg -y -v warning -i data/audio/2.mp3 -i ./results/v15/avatars/lisa/temp.mp4 ./results/v15/avatars/lisa/vid_output/audio_3.mp4
[mp3 @ 0x564515ae77c0] Estimating duration from bitrate, this may be inaccurate
result is save to ./results/v15/avatars/lisa/vid_output/audio_3.mp4

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk#


// 3rd try
preparation: False: 25 fps, 720 p, 2.6sec avatar video:: audio: 5 sec -> took: 1min 58 sec
(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# sh inference.sh v1.5 realtime
2025-08-11 06:34:55.334563: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-11 06:34:56.151825: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-11 06:35:01.313050: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loads checkpoint by local backend from path: ./models/dwpose/dw-ll_ucoco_384.pth
cuda start
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/utils/_contextlib.py:125: UserWarning: Decorating classes is deprecated and will be disabled in future versions. You should only decorate functions or methods. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.
  warnings.warn("Decorating classes is deprecated and will be disabled in "
An error occurred while trying to fetch models/sd-vae: Error no file named diffusion_pytorch_model.safetensors found in directory models/sd-vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
load unet model from ./models/musetalkV15/unet.pth
{'lisa': {'preparation': False, 'bbox_shift': 5, 'video_path': 'data/video/lisa26.mp4', 'audio_clips': {'audio_3': 'data/audio/3.mp3'}}}
reading images...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:01<00:00, 87.89it/s]
reading images...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:00<00:00, 391.92it/s]
Inferring using: data/audio/3.mp3
start inference
processing audio:data/audio/3.mp3 costs 6293.038368225098ms
129
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:07<00:00,  1.03s/it]
Total process time of 129 frames including saving images = 10.094971418380737s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa/temp.mp4
ffmpeg -y -v warning -i data/audio/3.mp3 -i ./results/v15/avatars/lisa/temp.mp4 ./results/v15/avatars/lisa/vid_output/audio_3.mp4
[mp3 @ 0x563c69e017c0] Estimating duration from bitrate, this may be inaccurate
result is save to ./results/v15/avatars/lisa/vid_output/audio_3.mp4

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk#

// 4th try
preparation: False: 25 fps, 720 p, 2.6sec avatar video:: audio(same audio as 3rd): 5 sec -> took: 1min 58 sec

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# sh inference.sh v1.5 realtime
2025-08-11 06:38:10.213532: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-11 06:38:10.278761: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-11 06:38:13.370917: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loads checkpoint by local backend from path: ./models/dwpose/dw-ll_ucoco_384.pth
cuda start
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/utils/_contextlib.py:125: UserWarning: Decorating classes is deprecated and will be disabled in future versions. You should only decorate functions or methods. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.
  warnings.warn("Decorating classes is deprecated and will be disabled in "
An error occurred while trying to fetch models/sd-vae: Error no file named diffusion_pytorch_model.safetensors found in directory models/sd-vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
load unet model from ./models/musetalkV15/unet.pth
{'lisa': {'preparation': False, 'bbox_shift': 5, 'video_path': 'data/video/lisa26.mp4', 'audio_clips': {'audio_3': 'data/audio/3.mp3'}}}
reading images...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:01<00:00, 99.94it/s]
reading images...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:00<00:00, 537.05it/s]
Inferring using: data/audio/3.mp3
start inference
processing audio:data/audio/3.mp3 costs 4172.593593597412ms
129
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:06<00:00,  1.02it/s]
Total process time of 129 frames including saving images = 9.34282374382019s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa/temp.mp4
ffmpeg -y -v warning -i data/audio/3.mp3 -i ./results/v15/avatars/lisa/temp.mp4 ./results/v15/avatars/lisa/vid_output/audio_3.mp4
[mp3 @ 0x55c7cd3357c0] Estimating duration from bitrate, this may be inaccurate
result is save to ./results/v15/avatars/lisa/vid_output/audio_3.mp4


(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk#





