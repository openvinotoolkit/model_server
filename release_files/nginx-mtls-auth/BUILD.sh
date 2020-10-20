#!/bin/bash -x
docker build  -f Dockerfile . \
        --build-arg http_proxy="http://proxy-mu.intel.com:911" --build-arg https_proxy=""http://proxy-mu.intel.com:912"" \
        --build-arg no_proxy="localhost,127.0.0.1,10.0.0.0/8,intel.com,.intel.com" \
        --build-arg BASE_IMAGE="openvino/model_server:rr-ovms" \
        -t openvino/model_server:rr-ovms-nginx-mtls
