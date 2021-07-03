## Docker installation 

All commands below should be run from repository root i.e., after you run `git clone git@github.com:iglu-contest/iglu_env.git && cd iglu_env`.

First, make sure you are using recent docker version. The installation was tested with version `20.10.6`. Also, make sure
you are using [nvidia-container-runtime](https://github.com/NVIDIA/nvidia-container-runtime) as default docker runtime.

To get the most recent image, pull that using 

```bash
docker pull iglucontest/env:latest
```

Alternatively, you can build images manually:

```bash
docker build -t iglu_base -f docker/base.Dockerfile .
docker build --build-arg BASE=iglu_base --network host -t iglu_env -f docker/iglu.Dockerfile .
```

## Testing installation

To test `iglu` in the container, run the following command from the root of cloned repo:

```bash
docker run --network host --rm -it -v $(pwd):/root/iglu_dev iglu python iglu_dev/test_env.py
```

You should see step counter followed by total reward of random agent.