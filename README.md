# Iglu Environment

## Installation

### 1. JDK 1.8 installation 

On Ubuntu/Debian:

```bash
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install openjdk-8-jdk
```

### 2. Git LFS installation

#### Windows
Download the windows installer from [here](https://github.com/git-lfs/git-lfs/releases)

Run the windows installer

Start a command prompt/or git for windows prompt and run ```git lfs install```

#### Debian and Ubuntu
Ubuntu 18.04, Debian 10, and newer versions of those OSes offer a git-lfs package. If you'd like to use that and don't need the latest version, skip step 1 below.

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

#### Mac OSX
You may need to brew update to get all the new formulas
brew install git-lfs
git lfs install


### 3. Install xvfb

#### Debian and Ubuntu
```bash
sudo apt-get install xvfb
```

### 4. IgluEnv installation 

```
git lfs clone git@github.com:iglu-contest/iglu_env.git && cd iglu_env
conda create -n iglu_env python=3.7
conda activate iglu_env
python setup.py install
```

#### And then test your installation
```
cd test && python test_env.py
```

