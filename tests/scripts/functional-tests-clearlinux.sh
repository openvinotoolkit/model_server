#!/bin/bash
set -ex

TEST_DIRS=tests
DOCKER_OVMS_TAG="ie-serving-clearlinux:latest"
export TESTS_SUFFIX="clearlinux"
export TESTS_PORTS="9100 9149 5600 5649"

make DOCKER_OVMS_TAG=${DOCKER_OVMS_TAG} docker_build_clearlinux

. .venv-jenkins/bin/activate

py.test ${TEST_DIRS}/functional/ -v --test_dir=/var/jenkins_home/test_ovms_models
