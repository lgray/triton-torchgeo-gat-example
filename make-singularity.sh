#!/bin/bash

TRITON_VERSIONS="20.06-py3 20.06-v1-py3"

for TRITON in $TRITON_VERSIONS
do
    echo $TRITON
    SINGULARITY_NOHTTPS=1 singularity build --fix-perms --sandbox tritonserver-${TRITON}-geometric docker-daemon://${USER}/tritonserver:${TRITON}-geometric
done
