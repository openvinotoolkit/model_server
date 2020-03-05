#!/bin/bash
set -ex

LD_LIBRARY_PATH=
LD_LIBRARY_PATH+=:/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib
LD_LIBRARY_PATH+=:/opt/intel/openvino/deployment_tools/inference_engine/external/mkltiny_lnx/lib
LD_LIBRARY_PATH+=:/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64
LD_LIBRARY_PATH+=:/opt/intel/openvino/deployment_tools/ngraph/lib

OPEN_VINO_DOCKER_IMAGE=openvino/ubuntu18_dev:2020.1
OVMS_TESTS_IMAGE=${OPEN_VINO_DOCKER_IMAGE}-ovms-tests

docker build \
    -t ${OVMS_TESTS_IMAGE} \
    -f `dirname $0`/Dockerfile.openvino \
    --build-arg OPEN_VINO_DOCKER_IMAGE=${OPEN_VINO_DOCKER_IMAGE} \
    `mktemp -d`

# Check python version in OV Docker Image
PYTHON_VERSION=`docker run --rm $OVMS_TESTS_IMAGE \
    /usr/bin/env python3 --version | awk -F"[ .]" '{ print $2"."$3 }'`

PYTHONPATH=/opt/intel/openvino/python/python${PYTHON_VERSION}

docker run -t --rm \
    -v $PWD:/mnt/ovms \
    -w /mnt/ovms \
    -e LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
    -e PYTHONPATH=${PYTHONPATH} \
    $OVMS_TESTS_IMAGE \
    bash -c "
    set -ex;
    source /opt/intel/openvino/bin/setupvars.sh
    make install;
    make unit;
    make coverage"
