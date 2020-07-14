# triton-torchgeo-gat-example
An example of running a pytorch-geometric graph attention model in nvidia triton.

Requirements:
- a recent docker installation (https://docs.docker.com/engine/install/)
- if you want to run the model on a gpu: 
  - install cuda 11 and its bundled driver (https://developer.nvidia.com/cuda-downloads)
  - install nvidia container runtimes (https://github.com/NVIDIA/nvidia-docker) 
    - NB: don't install nvidia-docker2!!!!

Use:
```
./build-images.sh # to build the various triton images
./start.sh # (or ./start_gpu.sh) to run dockerized tests
./cleanup.sh # to clean up the docker containers and networks
```
