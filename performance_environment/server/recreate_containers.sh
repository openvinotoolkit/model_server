#!/bin/bash

echo "Stopping ovms container"
docker stop ovms
echo "Removing ovms container"
docker rm ovms

echo "Stopping tfs container"
docker stop tfs
echo "Removing tfs container"
docker rm tfs

echo "Stopping minimal-server container"
docker stop minimal-server
echo "Removing minimal-server container"
docker rm minimal-server

echo "Starting ovms container.."
docker run -d --name ovms -v /opt/models/ovms/resnet:/resnet -p <ovms_port>>:<ovms_port> intelaipg/openvino-model-server /ie-serving-py/start_server.sh ie_serving model --model_path /resnet --model_name resnet --port <ovms_port>> --grpc_workers 32  --nireq 32 --target_device CPU --batch_size 1 --plugin_config "{\"CPU_THROUGHPUT_STREAMS\": \"CPU_THROUGHPUT_AUTO\"}"

echo "Starting tfs container.."
docker run -d -t --name tfs -p <tfs_port>:8500 -v /opt/models/tfs:/models -e MODEL_NAME=resnet tensorflow/serving

echo "Starting minimal-server container.."
docker run -d --name minimal-server -p <minimal_server_port>:9178 -e "http_proxy=$http_proxy" -e "https_proxy=$https_proxy" registry.tools.nervana.sclab.intel.com/cpp-experiments:v.1.1

echo "Checking the environment"
docker ps
echo "Environment is running"
