#!/bin/bash
set -ex

TEST_DIRS=tests

make docker_build_bin dldt_package_url=${OPENVINO_DOWNLOAD_LINK_2020_1}

. .venv-jenkins/bin/activate

py.test ${TEST_DIRS}/functional/ -v --test_dir=/var/jenkins_home/test_ovms_models
