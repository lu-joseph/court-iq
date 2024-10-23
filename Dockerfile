FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu20.04

# Set environment variables to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

EXPOSE 8888
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0

RUN python3 -m pip install --upgrade pip

RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

WORKDIR /root/
COPY . /root/
RUN pip install --ignore-installed -r requirements.txt

RUN apt-get update && apt-get install -y \
    screen \
    git \
    wget \
    unzip && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

CMD ["/root/container_entry.sh"]