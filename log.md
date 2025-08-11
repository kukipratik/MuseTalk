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

Preparation: False:
