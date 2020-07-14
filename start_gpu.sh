#!/bin/bash

docker network create gat_test_network

docker run -d --name gat_test_server \
       --gpus all \
       --network gat_test_network \
       --shm-size=1g --ulimit memlock=-1 \
       --ulimit stack=67108864 \
       -p8000:8000 -p8001:8001 -p8002:8002 \
       -v`pwd`/models:/models \
       -e LD_PRELOAD="libpyg/libtorchscatter.so libpyg/libtorchsparse.so" \
       local/gat_test_server:latest \
       tritonserver --model-repository=/models

sleep 45

docker run --name gat_test_client \
       --network gat_test_network \
       local/gat_test_client:latest \
       python /workspace/client.py -m gat_test -u gat_test_server:8001
