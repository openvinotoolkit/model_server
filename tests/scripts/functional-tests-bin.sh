#!/bin/bash
set -ex

TEST_DIRS=tests
DOCKER_OVMS_TAG="ie-serving-bin:latest"
export TESTS_SUFFIX="bin"
export PORTS_PREFIX="92 57"

make DOCKER_OVMS_TAG=${DOCKER_OVMS_TAG} docker_build_bin dldt_package_url=${OPENVINO_DOWNLOAD_LINK_2020_2}

. .venv-jenkins/bin/activate

py.test ${TEST_DIRS}/functional/ -v --test_dir=/var/jenkins_home/test_ovms_models-${TESTS_SUFFIX} --image ${DOCKER_OVMS_TAG} --ignore=${TEST_DIRS}/functional/test_ams_inference.py  --ignore=${TEST_DIRS}/functional/ams_schemas.py --ignore=${TEST_DIRS}/functional/test_single_model_vehicle*
