# Iglu Environment

The main documentation is available [here](https://iglu-contest.github.io/).

## Installation

### 1. JDK 1.8 installation 

#### On Ubuntu/Debian:

```bash
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install openjdk-8-jdk
```

#### On Mac:

```bash
brew tap AdoptOpenJDK/openjdk
brew install --cask adoptopenjdk8 #or `brew cask install adoptopenjdk8` for brew version < 3.
```

#### On Windows:

Please use [Windows installer](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html).

### 3. Install xvfb

By default `iglu` renders using virtual display using `xvfb` software renderer. 

#### Debian and Ubuntu
```bash
sudo apt-get install xvfb
```

### 4. IgluEnv installation 

`iglu` env requires python version `3.7` or newer. If you are using `conda` you can easily install that in local conda env:

```bash
conda create -n iglu_env python=3.7
conda activate iglu_env
```

You can install using pip: `TODO`

To install the package manually, do the following:

```
git clone git@github.com:iglu-contest/iglu_env.git && cd iglu_env
python setup.py install
```

#### And then test your installation
```
cd test && python test_env.py
```
### Suggested requirements for RL solution

```bash
conda env update --file conda_env.yml
```

### Docker installation 

To use `iglu` environment inside the docker container proceed to [Docker installation](docker/getting_started.md) section.

## Known Issues

### Java versions

You might have another java version installed (e.g. `openjdk-11`), rather than `openjdk-8-jdk`. Check the version by running `java --version` or `java -version`. 

To update default java runtime it's either `update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java` or `update-alternatives --config java`. The second one is preferable since it isn't conditioned on a particular path.
