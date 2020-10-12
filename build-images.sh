#!/bin/bash

# build necessary libraries and train the example model
source triton-version.sh
docker build -t ${USER}/tritonserver:${TRITON_VERSION}-geometric --build-arg SERVERBASE=${TRITON_VERSION} -f Dockerfile.build -m 16g .


# build container for running locally (uncomment if you want it)
#docker build -t local/gat_local_test -f Dockerfile.local .

docker run -d --name triton_artifacts \
       ${USER}/tritonserver:${TRITON_VERSION}-geometric

mkdir -p artifacts/models/gat_test/1

cp config.pbtxt artifacts/models/gat_test/
docker cp triton_artifacts:/torch_geometric/examples/model.pt artifacts/models/gat_test/1/

docker stop triton_artifacts
docker rm triton_artifacts
