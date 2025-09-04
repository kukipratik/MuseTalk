# Audio:
16kHz mono, audio fps = 50fps

* command:
ffmpeg -i data/audio/input.wav -ar 16000 -ac 1 data/audio/output.wav

# Video:
25 fps

* command:
ffmpeg -i data/video/input.mp4 -r 25 -c:v libx264 -pix_fmt yuv420p data/video/output.mp4

# Conclusion:
Video = 25 fps, Audio = 16 kHz mono