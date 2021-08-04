#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

from utils.parametrization import generate_test_object_name

try:
    # In user_config.py, user might export custom environment variables
    import user_config
except ImportError:
    pass

"""IMAGE - docker image name which should be used to run tests"""
image = os.environ.get("IMAGE", "openvino/model_server")

"""TEST_DIR -  location where models and test data should be copied from TEST_DIR_CACHE and deleted after tests"""
test_dir = os.environ.get("TEST_DIR", "/tmp/{}".format(generate_test_object_name(prefix='ovms_models')))

"""TEST_DIR_CACHE -  location where models and test data should be downloaded to and serve as cache for TEST_DIR"""
test_dir_cache = os.environ.get("TEST_DIR_CACHE", "/tmp/ovms_models_cache")

"""TEST_DIR_CLEANUP - if set to True, TEST_DIR directory will be removed
                      after tests execution"""
test_dir_cleanup = os.environ.get("TEST_DIR_CLEANUP", "True")
test_dir_cleanup = test_dir_cleanup.lower() == "true"

"""BUILD_LOGS -  path to dir where artifacts should be stored"""
artifacts_dir = os.environ.get("BUILD_LOGS", "")

"""START_CONTAINER_COMMAND - command to start ovms container"""
start_container_command = os.environ.get("START_CONTAINER_COMMAND", "")

"""CONTAINER_LOG_LINE - log line to check in container"""
container_log_line = os.environ.get("CONTAINER_LOG_LINE", "Server started on port")

"""OVMS_BINARY_PATH - path to ovms binary file; when specified, tests are executed against provided binary."""
ovms_binary_path = os.environ.get("OVMS_BINARY_PATH", None)

"""LOG_LEVEL - set log level """
log_level = os.environ.get("LOG_LEVEL", "INFO")

path_to_mount = os.path.join(test_dir, "saved_models")

path_to_mount_cache = os.path.join(test_dir_cache, "saved_models")

models_path = path_to_mount if ovms_binary_path else "/opt/ml"

"""TT_MINIO_IMAGE_NAME - Docker image for Minio"""
minio_image = os.environ.get("TT_MINIO_IMAGE_NAME", "minio/minio:latest")

""" TT_TARGET_DEVICE - one of "CPU", "GPU" """
target_device = os.environ.get("TT_TARGET_DEVICE", "CPU")

start_minio_container_command = 'server --address ":{}" /data'

container_minio_log_line = "Endpoint:"
