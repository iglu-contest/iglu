#FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
ENV CONDALINK "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

RUN apt-get update && apt-get install -y -q vim \
    git curl wget gcc make cmake g++ x11-xserver-utils \
    openjdk-8-jdk sudo xvfb ffmpeg zip unzip

ENV HOME=/root
RUN curl -so /root/miniconda.sh $CONDALINK \
 && chmod +x /root/miniconda.sh \
 && /root/miniconda.sh -b -p ~/miniconda \
 && rm /root/miniconda.sh
ENV PATH=/root/miniconda/bin:$PATH

RUN conda install conda-build \
 && conda create -y --name py37 python=3.7.3 \
 && conda clean -ya
ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/root/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
