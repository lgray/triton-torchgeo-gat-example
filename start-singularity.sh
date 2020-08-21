#!/bin/bash

singularity instance start \
	    -B /run/shm:/run/shm -B ./artifacts/models/:/models \
	    --hostname gattestserver --writable \
	    tritonserver-20.06-py3-geometric/ gat_test_server

singularity run instance://gat_test_server \
	    tritonserver --model-repository=/models >& gat_test_server.log &

sleep 2

singularity run -B `pwd`/client:/inputs \
	    docker://nvcr.io/nvidia/tritonserver:20.06-py3-clientsdk \
	    python /inputs/client.py -m gat_test -u localhost:8001
