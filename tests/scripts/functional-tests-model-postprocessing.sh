#!/bin/bash
set -ex

TEST_DIRS=tests
DOCKER_OVMS_TAG="ie-serving-apt-ubuntu:latest"
export TESTS_SUFFIX="apt-ubuntu"
export PORTS_PREFIX="91 56"

make DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG} docker_build_ams dldt_package_url=${OPENVINO_DOWNLOAD_LINK_2020_2}
make DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG}-clearlinux docker_build_ams_clearlinux dldt_package_url=${OPENVINO_DOWNLOAD_LINK_2020_2}

. .venv-jenkins/bin/activate

py.test ${TEST_DIRS}/functional/test_single_model_vehicle.py ${TEST_DIRS}/functional/test_single_model_vehicle_attributes.py  -v --test_dir=/var/jenkins_home/test_ovms_models-${TESTS_SUFFIX} --image ${DOCKER_OVMS_TAG}
py.test ${TEST_DIRS}/functional/test_single_model_vehicle.py ${TEST_DIRS}/functional/test_single_model_vehicle_attributes.py  -v --test_dir=/var/jenkins_home/test_ovms_models-${TESTS_SUFFIX} --image ${DOCKER_OVMS_TAG}-clearlinux

