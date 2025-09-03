# Audio:
ffmpeg -i input.mp3 -ac 1 -ar 16000 -c:a pcm_s16le audio_16k.wav

# Video:
25 fps

# Conclusion:
Video = 25 fps, Audio = 16 kHz mono