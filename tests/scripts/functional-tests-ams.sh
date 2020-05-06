#!/bin/bash
set -ex

TEST_DIRS=tests
DOCKER_IMAGE_TAG="ams:latest"
export TESTS_SUFFIX="bin"
export PORTS_PREFIX="92 57"

make DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG} docker_build_ams dldt_package_url=${OPENVINO_DOWNLOAD_LINK_2020_2}

. .venv-jenkins/bin/activate

set +e
NO_PROXY=localhost py.test ${TEST_DIRS}/functional/test_ams_inference.py -v --test_dir=/var/jenkins_home/test_ovms_models-${TESTS_SUFFIX} --image ${DOCKER_IMAGE_TAG}
exit_code=$?
exit ${exit_code}