ARG BASE
FROM ${BASE}

RUN pip install  numpy scipy matplotlib jupyterlab scikit-learn \ 
    gym "ray[rllib]==1.0.1" pandas tensorflow-gpu==2.3.0 \
    wandb moviepy transformers
RUN conda install pytorch=1.8.1 torchvision cudatoolkit=11.1 -c pytorch -c nvidia

