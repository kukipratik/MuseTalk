# Use an official CUDA base image with Python 3.10
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    FFMPEG_PATH=/usr/bin/ffmpeg

# System dependencies
RUN apt-get update && apt-get install -y \
    git wget ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (recommended for this repo)
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Accept Conda Terms of Service
RUN conda init bash && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment and install Python + PyTorch (CUDA 11.8)
RUN conda create -y -n musetalk python=3.10
SHELL ["conda", "run", "-n", "musetalk", "/bin/bash", "-c"]
RUN conda install -y pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Activate environment, install pip requirements, mmlab
COPY requirements.txt /workspace/requirements.txt
SHELL ["conda", "run", "-n", "musetalk", "/bin/bash", "-c"]
RUN pip install --upgrade pip \
 && pip install -r /workspace/requirements.txt \
 && pip install --no-cache-dir -U openmim \
 && mim install mmengine \
 && mim install "mmcv==2.0.1" \
 && mim install "mmdet==3.1.0" \
 && mim install "mmpose==1.1.0"

# Copy code and assets
COPY . /workspace
WORKDIR /workspace

# Download weights if not present (optional step, you can also mount volume)
# RUN bash download_weights.sh

# If you want to use app.py (Gradio), expose 7860
EXPOSE 7860

# Default: launch a bash shell in the conda env
CMD ["conda", "run", "-n", "musetalk", "bash"]
