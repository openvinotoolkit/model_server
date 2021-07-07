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

import shutil
import os
import pytest

import config
from model.models_information import Resnet, ResnetONNX, AgeGender
from object_model.server import Server


@pytest.fixture(scope="class")
def start_server_single_model(request):

    start_server_command_args = {"model_name": Resnet.name,
                                 "model_path": Resnet.model_path,
                                 "plugin_config": "\"{\\\"CPU_THROUGHPUT_STREAMS\\\": \\\"CPU_THROUGHPUT_AUTO\\\"}\""}
    container_name_infix = "test-single"

    # In this case, slower, non-default serialization method is used
    env_variables = ['SERIALIZATON=_prepare_output_as_AppendArrayToTensorProto']

    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command,
                    env_variables, target_device=config.target_device)
    return server.start()

@pytest.fixture(scope="class")
def start_server_single_model_onnx(request):

    start_server_command_args = {"model_name": ResnetONNX.name,
                                 "model_path": ResnetONNX.model_path,
                                 "plugin_config": "\"{\\\"CPU_THROUGHPUT_STREAMS\\\": \\\"CPU_THROUGHPUT_AUTO\\\"}\""}
    container_name_infix = "test-single-onnx"

    # In this case, slower, non-default serialization method is used
    env_variables = ['SERIALIZATON=_prepare_output_as_AppendArrayToTensorProto']

    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command,
                    env_variables, target_device=config.target_device)
    return server.start()

@pytest.fixture(scope="class")
def start_server_with_mapping(request):

    def delete_mapping_file():
        if os.path.exists(file_dst_path):
            os.remove(file_dst_path)

    request.addfinalizer(delete_mapping_file)

    file_dst_path = config.path_to_mount + '/age_gender/1/mapping_config.json'
    shutil.copyfile('tests/functional/mapping_config.json', file_dst_path)

    start_server_command_args = {"model_name": AgeGender.name,
                                 "model_path": AgeGender.model_path}
    container_name_infix = "test-2-out"
    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command,
                    target_device=config.target_device)
    return server.start()
