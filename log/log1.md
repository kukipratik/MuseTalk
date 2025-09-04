```bash
- infer called: avatar=lisa fps=25 prep=True video_path=data/video/lisa.mp4
*********************************
  creating avator: lisa
*********************************
preparing data materials ... ...
extracting landmarks...
reading images...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 244/244 [00:04<00:00, 50.00it/s]
get key_landmark and face bounding boxes with the bbox_shift: 5
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 244/244 [00:50<00:00,  4.83it/s]
********************************************bbox_shift parameter adjustment**********************************************************
Total frame:「244」 Manually adjust range : [ -14~13 ] , the current value: 5
*************************************************************************************************************************************
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 488/488 [01:08<00:00,  7.12it/s]
start inference
processing audio:/tmp/tmpbkf3i238/output.wav costs 5895.58ms
169
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:07<00:00,  1.15it/s]
Total process time of 169 frames including saving images = 17.742438077926636s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa/temp.mp4
ffmpeg -y -v warning -i /tmp/tmpbkf3i238/output.wav -i ./results/v15/avatars/lisa/temp.mp4 ./results/v15/avatars/lisa/vid_output/audio_upload.mp4
Guessed Channel Layout for Input Stream #0.0 : mono
result is save to ./results/v15/avatars/lisa/vid_output/audio_upload.mp4


- infer total: 196.60s | avatar=lisa fps=25 prep=True
- infer called: avatar=lisa fps=25 prep=False video_path=data/video/lisa.mp4
reading images...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 488/488 [00:09<00:00, 50.15it/s]
reading images...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 488/488 [00:01<00:00, 295.94it/s]
start inference
processing audio:/tmp/tmpggcryxv2/2.wav costs 44.44ms
257
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:07<00:00,  1.70it/s]
Total process time of 257 frames including saving images = 22.90308713912964s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa/temp.mp4
ffmpeg -y -v warning -i /tmp/tmpggcryxv2/2.wav -i ./results/v15/avatars/lisa/temp.mp4 ./results/v15/avatars/lisa/vid_output/audio_upload.mp4
Guessed Channel Layout for Input Stream #0.0 : mono
result is save to ./results/v15/avatars/lisa/vid_output/audio_upload.mp4


- infer total: 43.19s | avatar=lisa fps=25 prep=False
INFO:     100.64.0.28:52964 - "POST /infer HTTP/1.1" 200 OK
INFO:     100.64.0.28:36110 - "GET /docs HTTP/1.1" 200 OK
INFO:     100.64.0.28:36110 - "GET /openapi.json HTTP/1.1" 200 OK
- infer called: avatar=lisa fps=25 prep=False video_path=data/video/lisa.mp4
reading images...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 488/488 [00:08<00:00, 55.55it/s]
reading images...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 488/488 [00:01<00:00, 473.59it/s]
start inference
processing audio:/tmp/tmpr0hgsr7p/output (1).wav costs 29.31ms
416
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:10<00:00,  2.00it/s]
Total process time of 416 frames including saving images = 37.20725631713867s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa/temp.mp4
ffmpeg -y -v warning -i /tmp/tmpr0hgsr7p/output (1).wav -i ./results/v15/avatars/lisa/temp.mp4 ./results/v15/avatars/lisa/vid_output/audio_upload.mp4
sh: 1: Syntax error: "(" unexpected
result is save to ./results/v15/avatars/lisa/vid_output/audio_upload.mp4


- infer total: 54.42s | avatar=lisa fps=25 prep=False
INFO:     100.64.0.28:51292 - "POST /infer HTTP/1.1" 200 OK
- infer called: avatar=lisa fps=25 prep=False video_path=data/video/lisa.mp4
reading images...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 488/488 [00:09<00:00, 52.39it/s]
reading images...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 488/488 [00:01<00:00, 384.54it/s]
start inference
processing audio:/tmp/tmpug2w8ml_/output.wav costs 68.12ms
1220
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:25<00:00,  2.41it/s]
Total process time of 1220 frames including saving images = 106.33278250694275s
ffmpeg -y -v warning -r 25 -f image2 -i ./results/v15/avatars/lisa/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 ./results/v15/avatars/lisa/temp.mp4
ffmpeg -y -v warning -i /tmp/tmpug2w8ml_/output.wav -i ./results/v15/avatars/lisa/temp.mp4 ./results/v15/avatars/lisa/vid_output/audio_upload.mp4
Guessed Channel Layout for Input Stream #0.0 : mono
result is save to ./results/v15/avatars/lisa/vid_output/audio_upload.mp4


- infer total: 153.27s | avatar=lisa fps=25 prep=False
INFO:     100.64.0.34:36404 - "GET / HTTP/1.1" 404 Not Found
Connection to 100.65.14.1 closed.
Connection to ssh.runpod.io closed.
PS C:\Users\user\Desktop\voyage\ssh>
```