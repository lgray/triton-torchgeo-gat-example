#!/bin/bash

# build necessary libraries and train the example model
docker build -t local/gat_artifacts -f Dockerfile.build -m 8g .

# build container for running locally (uncomment if you want it)
#docker build -t local/gat_local_test -f Dockerfile.local .

docker run -d --name triton_artifacts \
       local/gat_artifacts:latest

mkdir -p artifacts/models/gat_test/1
mkdir -p artifacts/lib

cp config.pbtxt artifacts/models/gat_test/
docker cp triton_artifacts:/artifacts/model.pt artifacts/models/gat_test/1/
docker cp triton_artifacts:/artifacts/libtorchscatter.so artifacts/lib/
docker cp triton_artifacts:/artifacts/libtorchsparse.so artifacts/lib/
docker cp triton_artifacts:/artifacts/libtorchcluster.so artifacts/lib/
docker cp triton_artifacts:/artifacts/libtorchsplineconv.so artifacts/lib/

docker stop triton_artifacts
docker rm triton_artifacts
