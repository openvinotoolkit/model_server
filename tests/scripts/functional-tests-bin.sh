#!/bin/bash
set -ex

DOCKER_OVMS_TAG=$DOCKER_OVMS_TAG
TEST_DIRS=tests

make docker_build_bin dldt_package_url=${OPENVINO_DOWNLOAD_LINK_2020_1}

. .venv-jenkins/bin/activate

export PORT_RANGE=2
py.test ${TEST_DIRS}/functional/ -v --test_dir=/var/jenkins_home/test_models --image $DOCKER_OVMS_TAG
