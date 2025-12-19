FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel AS torch
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# pytorch
LABEL com.nvidia.volumes.needed=nvidia_driver
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*
COPY --from=torch /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH
ENV PYTORCH_VERSION=2.7.1

# set workdir
USER root
WORKDIR /root/workspace

# install dependencies
RUN apt-get update && DEBIAN_FRONTEND=nointeractive apt-get install -y --no-install-recommends \
        git \
        vim \
        wget \
        unzip \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# install opencv-python
ARG OPENCV_VERSION==4.5.5.64
RUN python3 -m pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN python3 -m pip install --no-cache-dir \
    numpy==1.26.4 \
    opencv-python==${OPENCV_VERSION} \
    opencv-python-headless==${OPENCV_VERSION} \
    opencv-contrib-python==${OPENCV_VERSION}

# set cuda env
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

### install ppl.cv
ARG PPLCV_VERSION=0.7.0
ARG PPLCV_REPO=https://github.com/openppl-public/ppl.cv.git
RUN git clone --depth 1 --branch v${PPLCV_VERSION} ${PPLCV_REPO} &&\
    cd ppl.cv &&\
    ./build.sh cuda && mv cuda-build/install ./ && rm -rf cuda-build
ENV pplcv_DIR=/root/workspace/ppl.cv/install/lib/cmake/ppl

### install tensorrt
ARG TENSORRT_URL
ARG TENSORRT_VERSION=8.6.1.6
RUN wget -c $TENSORRT_URL && \
    tar -zxvf /root/workspace/TensorRT-${TENSORRT_VERSION}*.tar.gz -C /root/workspace &&\
    rm -rf TensorRT-${TENSORRT_VERSION}*.tar.gz &&\
    ln -sf /root/workspace/TensorRT-${TENSORRT_VERSION} /root/workspace/TensorRT &&\
    cd /root/workspace/TensorRT && rm -rf data doc samples uff &&\
    export PY_VERSION=$(python3 -V | awk '{print $2}' | awk '{split($0, a, "."); print a[1]a[2]}') &&\
    python3 -m pip install ./python/tensorrt-*-cp${PY_VERSION}-none-linux_x86_64.whl
ENV TENSORRT_DIR=/root/workspace/TensorRT
ENV LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
ENV PATH=$TENSORRT_DIR/bin:$PATH

ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

# build mmdeploy
ARG MMDEPLOY_VERSION=main
ARG MMDEPLOY_REPO=https://github.com/open-mmlab/mmdeploy.git
ENV FORCE_CUDA="1"
RUN git clone --recursive -b $MMDEPLOY_VERSION --depth 1 $MMDEPLOY_REPO &&\
    bash mmdeploy/.github/scripts/linux/build.sh "cuda" "trt" \
        -Dpplcv_DIR=${pplcv_DIR} \
        -DTENSORRT_DIR=${TENSORRT_DIR}

ENV TORCH_CUDA_ARCH_LIST="8.6 8.9 9.0+PTX"
RUN python3 -m pip install --no-cache-dir ninja psutil pycuda onnxsim onnxruntime
RUN python3 -m pip install --no-cache-dir "mmcv>=2.0.0"
RUN cd mmdeploy &&\
    python3 -m pip install --no-cache-dir -r requirements.txt &&\
    python3 -m pip install -e .
ENV MMDeploy_DIR="/root/workspace/mmdeploy/build/install/lib/cmake/MMDeploy"
ENV LD_LIBRARY_PATH="/root/workspace/mmdeploy/build/lib:${LD_LIBRARY_PATH}"
ENV PATH="/root/workspace/mmdeploy/build/bin:${PATH}"
ENV PYTHONPATH="/root/workspace/mmdeploy/build/lib"
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# clear proxy
ENV HTTP_PROXY=
ENV HTTPS_PROXY=
