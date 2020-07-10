#!/bin/bash

docker rm -f gat_test_server
docker rm -f gat_test_client
docker network rm gat_test_network
