FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    build-essential \
    cmake \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory (mirrored to host via volume mount)
WORKDIR /workspace

# Copy and install requirements
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Verify CUDA is accessible from PyTorch
RUN python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Default command
CMD ["/bin/bash"]
