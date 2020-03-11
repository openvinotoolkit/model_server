#!/bin/bash
set -ex

TEST_DIRS=tests

make docker_build_apt_ubuntu

. .venv-jenkins/bin/activate

export PORT_RANGE=1
py.test ${TEST_DIRS}/functional/ -v --test_dir=/var/jenkins_home/test_models --image $DOCKER_OVMS_TAG
