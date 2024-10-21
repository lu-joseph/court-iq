FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu20.04

# Set environment variables to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    screen \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 -y && \
    rm -rf /var/lib/apt/lists/*

# USER root
# RUN apt-get update && apt-get install -y screen

RUN python3 -m pip install --upgrade pip

RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

WORKDIR /root/
COPY . /root/

RUN pip install --ignore-installed -r requirements.txt

EXPOSE 8888

SHELL ["/bin/bash", "-c"]

CMD ["/root/container_entry.sh"]