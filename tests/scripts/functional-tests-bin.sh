#!/bin/bash
set -ex

TEST_DIRS=tests
DOCKER_OVMS_TAG="ie-serving-bin:latest"
export TESTS_SUFFIX="bin"
export TESTS_PORTS="9050 9099 5550 5599"

make DOCKER_OVMS_TAG=${DOCKER_OVMS_TAG} docker_build_bin dldt_package_url=${OPENVINO_DOWNLOAD_LINK_2020_1}

. .venv-jenkins/bin/activate

py.test ${TEST_DIRS}/functional/ -v --test_dir=/var/jenkins_home/test_ovms_models
