Here, i am trying to do the inference for muse talk in real time for voyage realtime ai interview.

And below is the log for different scenerio:
1st log:
// Normal reference; avatar video: 7s, avatar audio 7s -> took time like 5 mins

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# sh inference.sh v1.5 normal
2025-08-11 05:17:15.313233: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-11 05:17:16.288441: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-11 05:17:24.173669: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loads checkpoint by local backend from path: ./models/dwpose/dw-ll_ucoco_384.pth
cuda start
Downloading: "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" to /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 85.7M/85.7M [00:01<00:00, 60.4MB/s]
An error occurred while trying to fetch models/sd-vae: Error no file named diffusion_pytorch_model.safetensors found in directory models/sd-vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
load unet model from ./models/musetalkV15/unet.pth
Loaded inference config: {'task_0': {'video_path': 'data/video/lisa.mp4', 'audio_path': 'data/audio/lisa.mp3'}}
Extracting landmarks... time-consuming operation
reading images...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [00:03<00:00, 60.27it/s]
get key_landmark and face bounding boxes with the default value
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [00:48<00:00,  3.96it/s]
********************************************bbox_shift parameter adjustment**********************************************************
Total frame:「192」 Manually adjust range : [ -11~11 ] , the current value: 0
*************************************************************************************************************************************
Number of frames: 192
Starting inference
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 22/22 [00:10<00:00,  2.06it/s]
Padding generated images to original video size
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 171/171 [00:32<00:00,  5.24it/s]
Video generation command: ffmpeg -y -v warning -r 24.0 -f image2 -i ./results/test/v15/lisa_lisa/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/test/v15/temp_lisa_lisa.mp4
Audio combination command: ffmpeg -y -v warning -i data/audio/lisa.mp3 -i ./results/test/v15/temp_lisa_lisa.mp4 ./results/test/v15/lisa_lisa.mp4
[mp3 @ 0x55fa31b8f7c0] Estimating duration from bitrate, this may be inaccurate
Results saved to ./results/test/v15/lisa_lisa.mp4
(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# python -m scripts.inference --inference_config configs/inference/test.yaml
2025-08-11 05:24:57.102729: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-11 05:24:57.163974: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-11 05:25:02.019712: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loads checkpoint by local backend from path: ./models/dwpose/dw-ll_ucoco_384.pth
cuda start
An error occurred while trying to fetch models/sd-vae: Error no file named diffusion_pytorch_model.safetensors found in directory models/sd-vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
load unet model from ./models/musetalkV15/unet.pth
Traceback (most recent call last):
  File "/workspace/miniconda3/envs/musetalk/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/workspace/miniconda3/envs/musetalk/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/workspace/MuseTalk/scripts/inference.py", line 276, in <module>
    main(args)
  File "/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/workspace/MuseTalk/scripts/inference.py", line 44, in main
    vae, unet, pe = load_all_model(
  File "/workspace/MuseTalk/musetalk/utils/utils.py", line 25, in load_all_model
    unet = UNet(
  File "/workspace/MuseTalk/musetalk/models/unet.py", line 36, in __init__
    with open(unet_config, 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: './models/musetalk/config.json'
(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk#


Preparation: True::

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# sh inference.sh v1.5 realtime
2025-08-11 05:31:43.366979: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-11 05:31:43.431578: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-11 05:31:47.266379: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loads checkpoint by local backend from path: ./models/dwpose/dw-ll_ucoco_384.pth
cuda start
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/utils/_contextlib.py:125: UserWarning: Decorating classes is deprecated and will be disabled in future versions. You should only decorate functions or methods. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.
  warnings.warn("Decorating classes is deprecated and will be disabled in "
An error occurred while trying to fetch models/sd-vae: Error no file named diffusion_pytorch_model.safetensors found in directory models/sd-vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
load unet model from ./models/musetalkV15/unet.pth
{'lisa': {'preparation': True, 'bbox_shift': 0, 'video_path': 'data/video/lisa.mp4', 'audio_clips': {'audio_0': 'data/audio/lisa.mp3'}}}
*********************************
  creating avator: lisa
*********************************
preparing data materials ... ...
extracting landmarks...
reading images...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [00:03<00:00, 49.30it/s]
get key_landmark and face bounding boxes with the default value
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [00:50<00:00,  3.82it/s]
********************************************bbox_shift parameter adjustment**********************************************************
Total frame:「192」 Manually adjust range : [ -11~11 ] , the current value: 0
*************************************************************************************************************************************
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 384/384 [01:06<00:00,  5.78it/s]
Inferring using: data/audio/lisa.mp3
start inference
processing audio:data/audio/lisa.mp3 costs 7354.91418838501ms
178
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:09<00:00,  1.03s/it]
Total process time of 178 frames including saving images = 18.61989974975586s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa/temp.mp4
ffmpeg -y -v warning -i data/audio/lisa.mp3 -i ./results/v15/avatars/lisa/temp.mp4 ./results/v15/avatars/lisa/vid_output/audio_0.mp4
[mp3 @ 0x55a25623e7c0] Estimating duration from bitrate, this may be inaccurate
result is save to ./results/v15/avatars/lisa/vid_output/audio_0.mp4


(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk#

Preparation: False: (1st Try)
(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# sh inference.sh v1.5 realtime
2025-08-11 05:40:49.399251: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-11 05:40:50.406352: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-11 05:40:58.162666: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loads checkpoint by local backend from path: ./models/dwpose/dw-ll_ucoco_384.pth
cuda start
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/utils/_contextlib.py:125: UserWarning: Decorating classes is deprecated and will be disabled in future versions. You should only decorate functions or methods. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.
  warnings.warn("Decorating classes is deprecated and will be disabled in "
An error occurred while trying to fetch models/sd-vae: Error no file named diffusion_pytorch_model.safetensors found in directory models/sd-vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
load unet model from ./models/musetalkV15/unet.pth
{'lisa': {'preparation': False, 'bbox_shift': -5, 'video_path': 'data/video/lisa.mp4', 'audio_clips': {'audio_0': 'data/audio/lisa.mp3'}}}
reading images...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 384/384 [00:07<00:00, 51.25it/s]
reading images...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 384/384 [00:01<00:00, 207.49it/s]
Inferring using: data/audio/lisa.mp3
start inference
processing audio:data/audio/lisa.mp3 costs 7654.072046279907ms
178
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:09<00:00,  1.04s/it]
Total process time of 178 frames including saving images = 18.93824338912964s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa/temp.mp4
ffmpeg -y -v warning -i data/audio/lisa.mp3 -i ./results/v15/avatars/lisa/temp.mp4 ./results/v15/avatars/lisa/vid_output/audio_0.mp4
[mp3 @ 0x55de4d9ea7c0] Estimating duration from bitrate, this may be inaccurate
result is save to ./results/v15/avatars/lisa/vid_output/audio_0.mp4

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk#

Preparation: False: (2nd Try)

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# sh inference.sh v1.5 realtime
2025-08-11 05:53:46.813197: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-11 05:53:47.598964: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-11 05:53:54.986485: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loads checkpoint by local backend from path: ./models/dwpose/dw-ll_ucoco_384.pth
cuda start
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/utils/_contextlib.py:125: UserWarning: Decorating classes is deprecated and will be disabled in future versions. You should only decorate functions or methods. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.
  warnings.warn("Decorating classes is deprecated and will be disabled in "
An error occurred while trying to fetch models/sd-vae: Error no file named diffusion_pytorch_model.safetensors found in directory models/sd-vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
load unet model from ./models/musetalkV15/unet.pth
{'lisa': {'preparation': False, 'bbox_shift': -7, 'video_path': 'data/video/idle0.mp4', 'audio_clips': {'audio_2': 'data/audio/2.mp3'}}}
reading images...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 384/384 [00:07<00:00, 48.57it/s]
reading images...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 384/384 [00:01<00:00, 221.77it/s]
Inferring using: data/audio/2.mp3
start inference
processing audio:data/audio/2.mp3 costs 9791.71347618103ms
204
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:08<00:00,  1.25it/s]
Total process time of 204 frames including saving images = 22.498761653900146s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa/temp.mp4
ffmpeg -y -v warning -i data/audio/2.mp3 -i ./results/v15/avatars/lisa/temp.mp4 ./results/v15/avatars/lisa/vid_output/audio_2.mp4
[mp3 @ 0x55af94fe97c0] Estimating duration from bitrate, this may be inaccurate
result is save to ./results/v15/avatars/lisa/vid_output/audio_2.mp4


(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk#

2nd log:
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

// 5th try
preparation: False: 25 fps, 720 p, 2.6sec avatar video:: audio: 3 sec -> took: 1min 29 sec

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# sh inference.sh v1.5 realtime
2025-08-11 07:16:17.638008: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-11 07:16:17.695391: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-11 07:16:21.520975: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loads checkpoint by local backend from path: ./models/dwpose/dw-ll_ucoco_384.pth
cuda start
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/utils/_contextlib.py:125: UserWarning: Decorating classes is deprecated and will be disabled in future versions. You should only decorate functions or methods. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.
  warnings.warn("Decorating classes is deprecated and will be disabled in "
An error occurred while trying to fetch models/sd-vae: Error no file named diffusion_pytorch_model.safetensors found in directory models/sd-vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
load unet model from ./models/musetalkV15/unet.pth
{'lisa': {'preparation': False, 'bbox_shift': 5, 'video_path': 'data/video/lisa26.mp4', 'audio_clips': {'audio_4': 'data/audio/4.mp3'}}}
reading images...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:01<00:00, 73.44it/s]
reading images...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:00<00:00, 291.93it/s]
Inferring using: data/audio/4.mp3
start inference
processing audio:data/audio/4.mp3 costs 6014.147996902466ms
84
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:05<00:00,  1.18s/it]
Total process time of 84 frames including saving images = 7.666594982147217s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa/temp.mp4
ffmpeg -y -v warning -i data/audio/4.mp3 -i ./results/v15/avatars/lisa/temp.mp4 ./results/v15/avatars/lisa/vid_output/audio_4.mp4
[mp3 @ 0x561dda98f7c0] Estimating duration from bitrate, this may be inaccurate
result is save to ./results/v15/avatars/lisa/vid_output/audio_4.mp4

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk#

// 6th try
preparation: False: 25 fps, 720 p, 2.6sec avatar video:: audio (same audio as 5th): 3 sec -> took: 1min 06 sec

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# sh inference.sh v1.5 realtime
2025-08-11 07:18:46.845494: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-11 07:18:46.903470: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-11 07:18:50.991558: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loads checkpoint by local backend from path: ./models/dwpose/dw-ll_ucoco_384.pth
cuda start
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/utils/_contextlib.py:125: UserWarning: Decorating classes is deprecated and will be disabled in future versions. You should only decorate functions or methods. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.
  warnings.warn("Decorating classes is deprecated and will be disabled in "
An error occurred while trying to fetch models/sd-vae: Error no file named diffusion_pytorch_model.safetensors found in directory models/sd-vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
load unet model from ./models/musetalkV15/unet.pth
{'lisa': {'preparation': False, 'bbox_shift': 5, 'video_path': 'data/video/lisa26.mp4', 'audio_clips': {'audio_4': 'data/audio/4.mp3'}}}
reading images...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:01<00:00, 104.54it/s]
reading images...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:00<00:00, 633.11it/s]
Inferring using: data/audio/4.mp3
start inference
processing audio:data/audio/4.mp3 costs 3136.5153789520264ms
84
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:05<00:00,  1.07s/it]
Total process time of 84 frames including saving images = 6.667635440826416s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa/temp.mp4
ffmpeg -y -v warning -i data/audio/4.mp3 -i ./results/v15/avatars/lisa/temp.mp4 ./results/v15/avatars/lisa/vid_output/audio_4.mp4
[mp3 @ 0x56536963c7c0] Estimating duration from bitrate, this may be inaccurate
result is save to ./results/v15/avatars/lisa/vid_output/audio_4.mp4

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk#

3rd log:
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
preparation: Fasle: 25 fps, 720 p, 1sec avatar video:: audio: 8 sec -> took: 1min 30 sec

** sorry forgot to keep log for this try **

//3rd try
preparation: False: 25 fps, 720 p, 1sec avatar video:: audio (same audio as 2nd try): 8 sec -> took: 1min 4 sec

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# sh inference.sh v1.5 realtime
2025-08-11 07:01:12.089126: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-11 07:01:12.142285: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-11 07:01:15.120971: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loads checkpoint by local backend from path: ./models/dwpose/dw-ll_ucoco_384.pth
cuda start
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/utils/_contextlib.py:125: UserWarning: Decorating classes is deprecated and will be disabled in future versions. You should only decorate functions or methods. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.
  warnings.warn("Decorating classes is deprecated and will be disabled in "
An error occurred while trying to fetch models/sd-vae: Error no file named diffusion_pytorch_model.safetensors found in directory models/sd-vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
load unet model from ./models/musetalkV15/unet.pth
{'lisa1': {'preparation': False, 'bbox_shift': 5, 'video_path': 'data/video/lisa1.mp4', 'audio_clips': {'audio_2': 'data/audio/2.mp3'}}}
reading images...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 127.63it/s]
reading images...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 580.06it/s]
Inferring using: data/audio/2.mp3
start inference
processing audio:data/audio/2.mp3 costs 3384.3555450439453ms
204
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:07<00:00,  1.42it/s]
Total process time of 204 frames including saving images = 12.621827125549316s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa1/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa1/temp.mp4
ffmpeg -y -v warning -i data/audio/2.mp3 -i ./results/v15/avatars/lisa1/temp.mp4 ./results/v15/avatars/lisa1/vid_output/audio_2.mp4
[mp3 @ 0x55c3aff7b7c0] Estimating duration from bitrate, this may be inaccurate
result is save to ./results/v15/avatars/lisa1/vid_output/audio_2.mp4

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk#

//4th try
preparation: False: 25 fps, 720 p, 1sec avatar video:: audio: 5 sec -> took: 1min 32 sec

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# sh inference.sh v1.5 realtime
2025-08-11 07:04:52.974423: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-11 07:04:53.030260: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-11 07:04:58.275984: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loads checkpoint by local backend from path: ./models/dwpose/dw-ll_ucoco_384.pth
cuda start
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/utils/_contextlib.py:125: UserWarning: Decorating classes is deprecated and will be disabled in future versions. You should only decorate functions or methods. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.
  warnings.warn("Decorating classes is deprecated and will be disabled in "
An error occurred while trying to fetch models/sd-vae: Error no file named diffusion_pytorch_model.safetensors found in directory models/sd-vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
load unet model from ./models/musetalkV15/unet.pth
{'lisa1': {'preparation': False, 'bbox_shift': 5, 'video_path': 'data/video/lisa1.mp4', 'audio_clips': {'audio_3': 'data/audio/3.mp3'}}}
reading images...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 93.70it/s]
reading images...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 495.00it/s]
Inferring using: data/audio/3.mp3
start inference
processing audio:data/audio/3.mp3 costs 5792.810440063477ms
129
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:07<00:00,  1.03s/it]
Total process time of 129 frames including saving images = 11.066500186920166s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa1/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa1/temp.mp4
ffmpeg -y -v warning -i data/audio/3.mp3 -i ./results/v15/avatars/lisa1/temp.mp4 ./results/v15/avatars/lisa1/vid_output/audio_3.mp4
[mp3 @ 0x55df2334a7c0] Estimating duration from bitrate, this may be inaccurate
result is save to ./results/v15/avatars/lisa1/vid_output/audio_3.mp4

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk#

//5th try
preparation: False: 25 fps, 720 p, 1sec avatar video:: audio: 3 sec -> took: 57 sec

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk# sh inference.sh v1.5 realtime
2025-08-11 07:10:17.167103: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-11 07:10:17.222718: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-11 07:10:20.538505: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loads checkpoint by local backend from path: ./models/dwpose/dw-ll_ucoco_384.pth
cuda start
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/utils/_contextlib.py:125: UserWarning: Decorating classes is deprecated and will be disabled in future versions. You should only decorate functions or methods. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.
  warnings.warn("Decorating classes is deprecated and will be disabled in "
An error occurred while trying to fetch models/sd-vae: Error no file named diffusion_pytorch_model.safetensors found in directory models/sd-vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
/workspace/miniconda3/envs/musetalk/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
load unet model from ./models/musetalkV15/unet.pth
{'lisa1': {'preparation': False, 'bbox_shift': 5, 'video_path': 'data/video/lisa1.mp4', 'audio_clips': {'audio_4': 'data/audio/4.mp3'}}}
reading images...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 100.73it/s]
reading images...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 690.23it/s]
Inferring using: data/audio/4.mp3
start inference
processing audio:data/audio/4.mp3 costs 3002.0804405212402ms
84
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:05<00:00,  1.06s/it]
Total process time of 84 frames including saving images = 6.605325937271118s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa1/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa1/temp.mp4
ffmpeg -y -v warning -i data/audio/4.mp3 -i ./results/v15/avatars/lisa1/temp.mp4 ./results/v15/avatars/lisa1/vid_output/audio_4.mp4
[mp3 @ 0x55dbc794e7c0] Estimating duration from bitrate, this may be inaccurate
result is save to ./results/v15/avatars/lisa1/vid_output/audio_4.mp4

(musetalk) root@bd124c3fc8d3:/workspace/MuseTalk#


well the inference latency decreased by 3rd log; but still it almost takes like more than 1 min; which is not good for rrealtime inference;

eventhough i have provided it with 24 gb vram; its not using all of its vrram power.
And, taking too much time;
at most 20 sec is tolerable...

And, for 2nd & 3rd log; this was the behaviour seen:
from executing "sh inference.sh v1.5 realtime" cmd to the terminal outputting "load unet model from ./models/musetalkV15/unet.pth" time took -> 0 sec to 36 sec on average

from the terminal outputting "load unet model from ./models/musetalkV15/unet.pth" to outputting "start inference" time took -> 36 sec to 50sec on average

And, the full process completed with output like "result is save to ./results/v15/avatars/lisa1/vid_output/audio_1.mp4" time took -> 50 sec to 1 min 8 sec