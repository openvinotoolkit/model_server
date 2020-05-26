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

try:
    # In user_config.py, user might export custom environment variables
    import user_config
except ImportError:
    pass

"""IMAGE - docker image name which should be used to run tests"""
image = os.environ.get("IMAGE", "ovms:latest")

"""TEST_DIR -  location where models and test data should be downloaded"""
test_dir = os.environ.get("TEST_DIR", "/tmp/ovms_models")

"""START_CONTAINER_COMMAND - command to start ovms container"""
start_container_command = os.environ.get("START_CONTAINER_COMMAND", "")

"""CONTAINER_LOG_LINE - log line to check in container"""
container_log_line = os.environ.get("CONTAINER_LOG_LINE", "Server started on port")

"""LOG_LEVEL - set log level """
log_level = os.environ.get("LOG_LEVEL", "INFO")

path_to_mount = os.path.join(test_dir, "saved_models")
