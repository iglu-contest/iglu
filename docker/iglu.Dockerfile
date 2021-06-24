ARG BASE
FROM ${BASE}

ADD . $HOME/iglu

RUN cd $HOME/iglu && python setup.py install
RUN rm -rf $HOME/iglu
WORKDIR $HOME
