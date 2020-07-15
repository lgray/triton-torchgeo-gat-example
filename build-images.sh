#!/bin/bash

# build the client container
docker build -t local/gat_test_client -f Dockerfile.client .

# build necessary libraries and train the example model
docker build -t local/gat_test_server -f Dockerfile.build .

# build container for running locally
docker build -t local/gat_local_test -f Dockerfile.local .

docker run -d --name triton_server_temp \
       --shm-size=1g --ulimit memlock=-1 \
       --ulimit stack=67108864 \
       -p8000:8000 -p8001:8001 -p8002:8002 \
       local/gat_test_server:latest

mkdir -p artifacts/models/gat_test/1
mkdir -p artifacts/lib

cp config.pbtxt artifacts/models/gat_test/
docker cp triton_server_temp:/opt/tritonserver/model.pt artifacts/models/gat_test/1/
docker cp triton_server_temp:/opt/tritonserver/libpyg/libtorchscatter.so artifacts/lib/
docker cp triton_server_temp:/opt/tritonserver/libpyg/libtorchsparse.so artifacts/lib/
docker cp triton_server_temp:/opt/tritonserver/libpyg/libtorchcluster.so artifacts/lib/
docker cp triton_server_temp:/opt/tritonserver/libpyg/libtorchsplineconv.so artifacts/lib/

docker stop triton_server_temp
docker rm triton_server_temp
