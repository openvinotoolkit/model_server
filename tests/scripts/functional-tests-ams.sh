#!/bin/bash
set -ex

TEST_DIRS=tests
DOCKER_IMAGE_TAG="ams:latest"
export TESTS_SUFFIX="bin"
export PORTS_PREFIX="92 57"

make DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG} docker_build_ams dldt_package_url=${OPENVINO_DOWNLOAD_LINK_2020_2}
make DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG}-clearlinux docker_build_ams_clearlinux dldt_package_url=${OPENVINO_DOWNLOAD_LINK_2020_2}

. .venv-jenkins/bin/activate

NO_PROXY=localhost py.test ${TEST_DIRS}/functional/test_ams_inference.py -v --test_dir=/var/jenkins_home/test_ovms_models-${TESTS_SUFFIX} --image ${DOCKER_IMAGE_TAG}
NO_PROXY=localhost py.test ${TEST_DIRS}/functional/test_ams_inference.py -v --test_dir=/var/jenkins_home/test_ovms_models-${TESTS_SUFFIX} --image ${DOCKER_IMAGE_TAG}-clearlinux
