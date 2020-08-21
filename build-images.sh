#!/bin/bash

# build necessary libraries and train the example model
export TRITON_VERSIONS="20.06-v1-py3 20.06-py3"
for TRITON in $TRITON_VERSIONS
do
    docker build -t ${USER}/tritonserver:${TRITON}-geometric --build-arg SERVERBASE=${TRITON} -f Dockerfile.build -m 16g .
done


# build container for running locally (uncomment if you want it)
#docker build -t local/gat_local_test -f Dockerfile.local .

docker run -d --name triton_artifacts \
       ${USER}/tritonserver:20.06-py3-geometric

mkdir -p artifacts/models/gat_test/1

cp config.pbtxt artifacts/models/gat_test/
docker cp triton_artifacts:/torch_geometric/examples/model.pt artifacts/models/gat_test/1/

docker stop triton_artifacts
docker rm triton_artifacts
