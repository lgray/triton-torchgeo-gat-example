#!/bin/bash

source triton-version.sh

singularity instance stop gat_test_server

rm tritonserver-${TRITON_VERSION}-geometric/opt/tritonserver/lib/libnvidia-ml.so.1
