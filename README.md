# triton-torchgeo-gat-example
An example of running a pytorch-geometric graph attention model in nvidia triton.

Requirements:
- a recent docker installation
- optionally: nvidia docker runtimes

Use:
```
./build-images.sh # to build the various triton images
./start.sh # (or ./start_gpu.sh) to run dockerized tests
./cleanup.sh # to clean up the docker containers and networks
```