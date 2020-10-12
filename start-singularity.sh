#!/bin/bash

source triton-version.sh

singularity instance start \
	    -B /run/shm:/run/shm -B ./artifacts/models/:/models \
	    --hostname gattestserver --writable \
	    tritonserver-${TRITON_VERSION}-geometric/ gat_test_server

singularity run instance://gat_test_server \
	    tritonserver --model-repository=/models >& gat_test_server.log &

sleep 2

singularity run -B `pwd`/client:/inputs \
	    docker://nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-clientsdk \
	    python /inputs/client.py -m gat_test -u localhost:8001
