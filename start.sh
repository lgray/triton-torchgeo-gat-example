#!/bin/bash

docker network create gat_test_network

docker run -d --name gat_test_server \
       --network gat_test_network \
       --shm-size=1g --ulimit memlock=-1 \
       --ulimit stack=67108864 \
       -p8000:8000 -p8001:8001 -p8002:8002 \
       -v`pwd`/artifacts:/inputs \
       -e LD_LIBRARY_PATH="/opt/tritonserver/lib/pytorch:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64" \
       -e LD_PRELOAD="/inputs/lib/libtorchscatter.so /inputs/lib/libtorchsparse.so" \
       nvcr.io/nvidia/tritonserver:20.06-v1-py3 \
       tritonserver --model-repository=/inputs/models

