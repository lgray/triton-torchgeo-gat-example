#!/bin/bash

source triton-version.sh

SINGULARITY_NOHTTPS=1 singularity build --fix-perms --sandbox tritonserver-${TRITON_VERSION}-geometric docker-daemon://${USER}/tritonserver:${TRITON_VERSION}-geometric
