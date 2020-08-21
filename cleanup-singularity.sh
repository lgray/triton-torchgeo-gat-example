#!/bin/bash

singularity instance stop gat_test_server

rm tritonserver-20.06-py3-geometric/opt/tritonserver/lib/libnvidia-ml.so.1
