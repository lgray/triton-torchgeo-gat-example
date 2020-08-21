#!/bin/bash

docker network create gat_test_network

docker run -d --name gat_test_server \
       --network gat_test_network \
       --shm-size=1g --ulimit memlock=-1 \
       --ulimit stack=67108864 \
       -p8000:8000 -p8001:8001 -p8002:8002 \
       -v`pwd`/artifacts/models:/models \
       ${USER}/tritonserver:20.06-py3-geometric \
       tritonserver --model-repository=/models

sleep 2

docker run --name gat_test_client \
       --network gat_test_network \
       -v`pwd`/client:/inputs \
       nvcr.io/nvidia/tritonserver:20.06-py3-clientsdk \
       python /inputs/client.py -m gat_test -u gat_test_server:8001
