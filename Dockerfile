#FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
ENV CONDALINK "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

RUN apt-get update && apt-get install -y -q vim \
    git curl wget gcc make cmake g++ x11-xserver-utils \
    openjdk-8-jdk sudo xvfb ffmpeg zip unzip

ARG USER_ID

RUN adduser --disabled-password -u ${USER_ID} --gecos '' --shell /bin/bash miner 
RUN echo "miner ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER miner
ENV HOME=/home/miner
RUN chmod 777 /home/miner

RUN curl -so ~/miniconda.sh $CONDALINK \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/miner/miniconda/bin:$PATH

RUN conda install conda-build \
 && conda create -y --name py37 python=3.7.3 \
 && conda clean -ya
ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/home/miner/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

#ADD .netrc /home/miner/
#ADD malmo.zip /home/miner/malmo

RUN conda install -c crowdai malmo
RUN pip install  numpy scipy matplotlib jupyterlab scikit-learn \ 
    minerl gym "ray[rllib]==1.0.1" pandas tensorflow-gpu==2.3.0 chainerrl \
    marlo wandb moviepy transformers
RUN conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
#RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

ADD . /home/miner/compet
RUN sudo chown -R miner /home/miner/compet
# RUN cd /home/miner/malmo \
#     && unzip /home/miner/malmo/malmo.zip
