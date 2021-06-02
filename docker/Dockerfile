ARG BASE
FROM ${BASE}

ADD . /home/builder/iglu

RUN sudo chown -R builder /home/builder/iglu
RUN cd /home/builder/iglu && python setup.py install
RUN mkdir /home/builder/iglu_dev
WORKDIR /home/builder

