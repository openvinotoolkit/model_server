#
# Copyright (c) 2019 Intel Corporation
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

import pytest
import shutil

import config
from model.models_information import Resnet
from utils.parametrization import get_tests_suffix
from object_model.server import Server


@pytest.fixture(scope="function")
def start_server_update_flow_latest(request):

    update_test_dir = config.path_to_mount + '/update-{}/'.format(get_tests_suffix())
    # ensure model dir is empty before starting OVMS
    shutil.rmtree(update_test_dir, ignore_errors=True)

    start_server_command_args = {"model_name": Resnet.name,
                                 "model_path": "/opt/ml/update-{}".format(get_tests_suffix()),
                                 "grpc_workers": 1,
                                 "nireq": 1}
    container_name_infix = "test-update-latest"
    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command,
                    target_device=config.target_device)
    return server.start()


@pytest.fixture(scope="function")
def start_server_update_flow_specific(request):

    update_test_dir = config.path_to_mount + '/update-{}/'.format(get_tests_suffix())
    # ensure model dir is empty before starting OVMS
    shutil.rmtree(update_test_dir, ignore_errors=True)

    start_server_command_args = {"model_name": Resnet.name,
                                 "model_path": "/opt/ml/update-{}".format(get_tests_suffix()),
                                 "model_version_policy": '\'{"specific": { "versions":[1, 3, 4] }}\''}
    container_name_infix = "test-update-specific"
    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command,
                    target_device=config.target_device)
    return server.start()
