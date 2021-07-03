Installation
============

.. _Windows installer: https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html

.. _nvidia-container-runtime: https://github.com/NVIDIA/nvidia-container-runtime

Global
------

Install JDK 1.8
***************

On Ubuntu/Debian:

.. code-block:: bash

   sudo add-apt-repository ppa:openjdk-r/ppa
   sudo apt-get update
   sudo apt-get install openjdk-8-jdk

On Mac:

.. code-block:: bash

   brew tap AdoptOpenJDK/openjdk
   brew cask install adoptopenjdk8


On Windows:

Please use `Windows installer`_.

Install xvfb
************

By default `iglu` renders using virtual display using `xvfb` software renderer. 

Debian and Ubuntu

.. code-block:: bash

   sudo apt-get install xvfb

install IGLU environment
************************

`iglu` env requires python version `3.7` or newer. If you are using `conda` you can easily install that in local conda env:

.. code-block:: bash

   conda create -n iglu_env python=3.7
   conda activate iglu_env

You can install using pip: ``pip install iglu``

To install the package manually, do the following:

.. code-block:: bash

   git clone git@github.com:iglu-contest/iglu_env.git && cd iglu_env
   python setup.py install


Docker
------

All commands below should be run from repository root i.e., after you run ``git clone git@github.com:iglu-contest/iglu_env.git && cd iglu_env``.

First, make sure you are using recent docker version. The installation was tested with version `20.10.6`. Also, make sure
you are using nvidia-container-runtime_ as default docker runtime.

To get the most recent image, pull that using 

.. code-block:: bash

   docker pull iglucontest/env:latest

Alternatively, you can build images manually:

.. code-block:: bash
   
   docker build -t iglu_base -f docker/base.Dockerfile .
   docker build --build-arg BASE=iglu_base --network host -t iglu_env -f docker/iglu.Dockerfile .


Testing installation
********************

To test `iglu` in the container, run the following command from the root of cloned repo:

.. code-block:: bash

   docker run --network host --rm -it -v $(pwd):/root/iglu_dev iglu python iglu_dev/test_env.py


You should see step counter followed by total reward of random agent.