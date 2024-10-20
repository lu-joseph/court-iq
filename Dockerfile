FROM rapidsai/rapidsai:cuda11.8-runtime-ubuntu20.04-py3.9

# Set environment variables to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    curl \
    git \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    # && add-apt-repository ppa:deadsnakes/ppa \
    # && apt-get update && apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Set the working directory
WORKDIR /rapids/notebooks/host/

COPY . /rapids/notebooks/host/

RUN pip install --ignore-installed -r /rapids/notebooks/host/requirements.txt

SHELL ["/bin/bash", "-c"]

# # Default command
CMD ["/bin/bash"]