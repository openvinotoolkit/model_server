#!/bin/bash
set -ex

TEST_DIRS=tests
DOCKER_OVMS_TAG="ie-serving-clearlinux:latest"
export TESTS_SUFFIX="clearlinux"
export PORTS_PREFIX="93 58"

make DOCKER_OVMS_TAG=${DOCKER_OVMS_TAG} docker_build_clearlinux

. .venv-jenkins/bin/activate

py.test ${TEST_DIRS}/functional/ -v --test_dir=/var/jenkins_home/test_ovms_models-${TESTS_SUFFIX} --image ${DOCKER_OVMS_TAG} --ignore=${TEST_DIRS}/functional/test_ams_inference.py  --ignore=${TEST_DIRS}/functional/ams_schemas.py --ignore=${TEST_DIRS}/functional/test_single_model_vehicle*
