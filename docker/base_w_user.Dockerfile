#FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
ENV CONDALINK "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

RUN apt-get update && apt-get install -y -q vim \
    git curl wget gcc make cmake g++ x11-xserver-utils \
    openjdk-8-jdk sudo xvfb ffmpeg zip unzip

ARG USER_ID

RUN adduser --disabled-password -u ${USER_ID} --gecos '' --shell /bin/bash builder 
RUN echo "builder ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER builder
ENV HOME=/home/builder
RUN chmod 777 /home/builder

RUN curl -so ~/miniconda.sh $CONDALINK \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/builder/miniconda/bin:$PATH

RUN conda install conda-build \
 && conda create -y --name py37 python=3.7.3 \
 && conda clean -ya
ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/home/builder/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH



