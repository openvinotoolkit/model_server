#!/bin/bash -x
docker build  -f Dockerfile . \
        --build-arg http_proxy="$http_proxy" --build-arg https_proxy="$https_proxy" \
        --build-arg no_proxy="$no_proxy" \
        --build-arg BASE_IMAGE="openvino/model_server:ovms" \
        -t openvino/model_server:nginx-mtls
