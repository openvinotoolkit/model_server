#!/bin/bash
set -ex

TEST_DIRS=tests
DOCKER_OVMS_TAG="ie-serving-bin:latest"
export TESTS_SUFFIX="bin"
export PORTS_PREFIX="92 57"

mkdir -p extras/models
make DOCKER_OVMS_TAG=${DOCKER_OVMS_TAG} docker_build_ams dldt_package_url=${OPENVINO_DOWNLOAD_LINK_2020_2}

. .venv-jenkins/bin/activate

# Start mock server
# TODO: remove this section and related changes after ams wrapper is ready for functional tests
python ${TEST_DIRS}/functional/utils/ams_mock_server.py &
mock_server_pid=$!

set +e
NO_PROXY=localhost py.test ${TEST_DIRS}/functional/test_single_model_vehicle* -v --test_dir=/var/jenkins_home/test_ovms_models-${TESTS_SUFFIX} --image ${DOCKER_OVMS_TAG}
exit_code=$?
kill ${mock_server_pid}
exit ${exit_code}
