#!/bin/bash
set -ex

TEST_DIRS=tests

make docker_build_clearlinux

. .venv-jenkins/bin/activate

export PORT_RANGE=3
py.test ${TEST_DIRS}/functional/ -v --test_dir=/var/jenkins_home/test_models --image "$DOCKER_OVMS_TAG"
