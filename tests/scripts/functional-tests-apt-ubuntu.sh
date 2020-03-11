#!/bin/bash
set -ex

TEST_DIRS=tests
DOCKER_OVMS_TAG="ie-serving-apt-ubuntu:latest"
PORT_RANGE=1

make DOCKER_OVMS_TAG=${DOCKER_OVMS_TAG} docker_build_apt_ubuntu

. .venv-jenkins/bin/activate

py.test ${TEST_DIRS}/functional/ -v --test_dir=/var/jenkins_home/test_models --image ${DOCKER_OVMS_TAG}
