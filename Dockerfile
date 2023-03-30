## Use an existing docker image as your base, example:
#FROM 747303060528.dkr.ecr.us-east-2.amazonaws.com/mstar-gitlab:MStarLogger as base
#FROM nvidia/cuda:11.3.1-devel-ubuntu20.04 as base
#FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SHELL=/bin/bash

RUN rm -rf /var/lib/apt/lists/*; \
    apt-get purge -y --auto-remove; \
    apt-get autoremove; \
    apt-get clean;

RUN apt --fix-missing update
RUN apt update
RUN apt install -f

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -f -y \
        build-essential autoconf libtool cmake ninja-build fuse iproute2 \
        libcudnn8 libcudnn8-dev \
        libzstd-dev wget git unzip python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN TORCH_CUDA_ARCH_LIST="8.0 7.5 7.0" FORCE_CUDA=1 python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
#RUN TORCH_CUDA_ARCH_LIST="8.0 7.5 7.0" FORCE_CUDA=1 python3 -m pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

## Fix for numpy float runtime error
RUN pip install "numpy<1.24"

RUN mkdir -p /usr/local/src/
WORKDIR /usr/local/src
RUN git clone https://github.com/anxuthu/apex.git
WORKDIR /usr/local/src/apex
RUN TORCH_CUDA_ARCH_LIST="8.0 7.5 7.0" FORCE_CUDA=1 python3 -m pip install -r requirements.txt
RUN TORCH_CUDA_ARCH_LIST="8.0 7.5 7.0" FORCE_CUDA=1 python3 -m pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

WORKDIR /usr/local/src/
RUN TORCH_CUDA_ARCH_LIST="8.0 7.5 7.0" FORCE_CUDA=1 python3 -m pip install regex ninja nltk pybind11 six
RUN git clone https://<access_token>@github.com/ParamsRaman/megatron.git
