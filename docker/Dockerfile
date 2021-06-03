ARG BASE
FROM ${BASE}

ADD . $HOME/iglu

RUN cd $HOME/iglu && python setup.py install
RUN mkdir $HOME/iglu_dev
WORKDIR $HOME
