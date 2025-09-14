FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglfw3 \
    libglew-dev \
    patchelf \
    gcc \
    python3.8-dev \
    unzip \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    vim \
    openssh-server \
    swig

RUN pip3 install --upgrade pip setuptools wheel

RUN pip3 install  \
    mujoco \
    gymnasium \
    "gymnasium[mujoco]" \
    mo-gymnasium[all] \
    matplotlib \
    pandas \
    fire \
    termcolor \
    python-dateutil \
    pygame \
    cloudpickle \
    pyparsing \
    cycler \
    kiwisolver \
    wandb \
    git+https://github.com/GwangPyo/safety-gymnasium_311_compat \
    pydantic \
    sentry_sdk \
    glfw \
    xmltodict \
    imageio \
    gymnasium_robotics \
    pynvml

RUN pip3 install stable-baselines3 --no-deps

