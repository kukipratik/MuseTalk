# Folder Stucture:
PS C:\Users\user\Desktop\voyage\MuseTalk> tree /f
Folder PATH listing
Volume serial number is 72F3-CD13
C:.
│   .gitignore
│   app.py
│   download_weights.bat
│   download_weights.sh
│   entrypoint.sh
│   inference.sh
│   LICENSE
│   README.md
│   requirements.txt
│   test_ffmpeg.py
│   train.py
│   train.sh
│   
├───assets
│   │   BBOX_SHIFT.md
│   │   
│   ├───demo
│   │   ├───man
│   │   │       man.png
│   │   │       
│   │   ├───monalisa
│   │   │       monalisa.png
│   │   │       
│   │   ├───musk
│   │   │       musk.png
│   │   │       
│   │   ├───sit
│   │   │       sit.jpeg
│   │   │       
│   │   ├───sun1
│   │   │       sun.png
│   │   │       
│   │   ├───sun2
│   │   │       sun.png
│   │   │       
│   │   ├───video1
│   │   │       video1.png
│   │   │       
│   │   └───yongen
│   │           yongen.jpeg
│   │
│   └───figs
│           gradio.png
│           gradio_2.png
│           landmark_ref.png
│           musetalk_arc.jpg
│
├───configs
│   ├───inference
│   │       realtime.yaml
│   │       test.yaml
│   │
│   └───training
│           gpu.yaml
│           preprocess.yaml
│           stage1.yaml
│           stage2.yaml
│           syncnet.yaml
│
├───data
│   ├───audio
│   │       1.wav
│   │       2.wav
│   │       3.wav
│   │       eng.wav
│   │       sun.wav
│   │       yongen.wav
│   │
│   └───video
│           lisa.mp4
│           sun.mp4
│           yongen.mp4
│
├───musetalk
│   ├───data
│   │       audio.py
│   │       dataset.py
│   │       sample_method.py
│   │
│   ├───loss
│   │       basic_loss.py
│   │       conv.py
│   │       discriminator.py
│   │       resnet.py
│   │       syncnet.py
│   │       vgg_face.py
│   │
│   ├───models
│   │       syncnet.py
│   │       unet.py
│   │       vae.py
│   │
│   ├───utils
│   │   │   audio_processor.py
│   │   │   blending.py
│   │   │   preprocessing.py
│   │   │   training_utils.py
│   │   │   utils.py
│   │   │   __init__.py
│   │   │
│   │   ├───dwpose
│   │   │       default_runtime.py
│   │   │       rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py
│   │   │
│   │   ├───face_detection
│   │   │   │   api.py
│   │   │   │   models.py
│   │   │   │   README.md
│   │   │   │   utils.py
│   │   │   │   __init__.py
│   │   │   │
│   │   │   └───detection
│   │   │       │   core.py
│   │   │       │   __init__.py
│   │   │       │
│   │   │       └───sfd
│   │   │               bbox.py
│   │   │               detect.py
│   │   │               net_s3fd.py
│   │   │               sfd_detector.py
│   │   │               __init__.py
│   │   │
│   │   └───face_parsing
│   │           model.py
│   │           resnet.py
│   │           __init__.py
│   │
│   └───whisper
│       │   audio2feature.py
│       │
│       └───whisper
│           │   audio.py
│           │   decoding.py
│           │   model.py
│           │   tokenizer.py
│           │   transcribe.py
│           │   utils.py
│           │   __init__.py
│           │   __main__.py
│           │
│           ├───assets
│           │   │   mel_filters.npz
│           │   │
│           │   ├───gpt2
│           │   │       merges.txt
│           │   │       special_tokens_map.json
│           │   │       tokenizer_config.json
│           │   │       vocab.json
│           │   │
│           │   └───multilingual
│           │           added_tokens.json
│           │           merges.txt
│           │           special_tokens_map.json
│           │           tokenizer_config.json
│           │           vocab.json
│           │
│           └───normalizers
│                   basic.py
│                   english.json
│                   english.py
│                   __init__.py
│
└───scripts
        inference.py
        preprocess.py
        realtime_inference.py
        __init__.py

PS C:\Users\user\Desktop\voyage\MuseTalk> 

# Notes:
I will be using RTX A5000 for testing (24 GB vram, 25 GB RAM, 9vCPU, 10 max).
As per research report, we will do this:
1. Run **muse_model** in GPU (and lock them), and for whisper_tiny, we will go with cpu (in order to avoid context switch problem)
2. On starting do this "@app.on_event("startup") \n def load_models():" for warming up models
3. expose "/prepare_avatar" for preparing avatar (always prep = true)
4. Expose "POST /lip_sync" for realtime inference (always prep = false), if avatar id not found return error.
5. We will have 3 avatars prepared "lisa", "lucy" and "mark" for now.
6. For now, we won't do streaming; we will just retrun full video for now 
7. fast api worker, always 1.
8. we will also cache the avatar assets for realtime "/lip_sync" (like said in report: "5. Persisting and Caching")
9. Don't prepare the character server start or something like that; i will prepare avatars myselfs by hitting "/prepare_avatar" api.
10. Instead of calling " -> its config path will be "config_path="configs/inference/realtime.yaml"" these yaml file; we will do work with params, we will accept and pass every required fields from params instead. (batch size and everything needed respectively for "POST /lip_sync" and "/prepare_avatar" api)
11. Finally, add timer "POST /lip_sync" and "/prepare_avatar" so that i can know how much time it took to give me response. Give proper log: 1st when api is hit (give api inferr hit with this param log), 2nd when returning the response.

# Final:
Now, real my requirements, research report and lets work to setup fast api server brother