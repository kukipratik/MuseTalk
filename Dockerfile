# Use official CUDA base image with Python 3.10 (CUDA 11.8 for torch==2.0.1+cu118)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    FFMPEG_PATH=/workspace/ffmpeg-4.4-amd64-static/ffmpeg

# System dependencies (includes gcc for pip packages that need compiling)
RUN apt-get update && apt-get install -y \
    git wget ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Accept Conda Terms of Service
RUN conda init bash && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment "MuseTalk" with Python 3.10
RUN conda create -y -n MuseTalk python=3.10

# Always run next steps in MuseTalk env
SHELL ["conda", "run", "-n", "MuseTalk", "/bin/bash", "-c"]

# Upgrade pip, install PyTorch (CUDA 11.8 wheels), TorchVision, and TorchAudio via pip
RUN pip install --upgrade pip && \
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# Install OpenMMLab ecosystem (mmcv, mmdet, mmengine, mmpose, openmim)
RUN pip install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmcv==2.0.1" && \
    mim install "mmdet==3.1.0" && \
    mim install "mmpose==1.1.0"

# (Optional) Download or copy static ffmpeg binary if you want a self-contained ffmpeg:
# Uncomment these lines and add your ffmpeg binary to the image if needed
# RUN wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -O /tmp/ffmpeg.tar.xz \
#     && mkdir -p /workspace/ffmpeg-4.4-amd64-static \
#     && tar -xf /tmp/ffmpeg.tar.xz --strip-components=1 -C /workspace/ffmpeg-4.4-amd64-static \
#     && rm /tmp/ffmpeg.tar.xz

# Set FFMPEG_PATH environment variable (for python scripts)
ENV FFMPEG_PATH=/workspace/ffmpeg-4.4-amd64-static/ffmpeg

# Copy code and assets
COPY . /workspace
WORKDIR /workspace

# Expose Gradio or API port (if needed)
EXPOSE 7860

# Default: launch a bash shell in the MuseTalk conda environment
CMD ["conda", "run", "-n", "MuseTalk", "bash"]
