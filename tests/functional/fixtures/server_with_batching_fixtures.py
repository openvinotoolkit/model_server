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

import config
from model.models_information import ResnetBS8, AgeGender
from object_model.server import Server

from functional.config import target_device


@pytest.fixture(scope="class")
def start_server_batch_model(request):

    start_server_command_args = {"model_name": ResnetBS8.name,
                                 "model_path": ResnetBS8.model_path}
    container_name_infix = "test-batch"
    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command,
                    target_device=config.target_device)
    return server.start()


@pytest.fixture(scope="class")
def start_server_batch_model_2out(request):

    start_server_command_args = {"model_name": AgeGender.name,
                                 "model_path": AgeGender.model_path}
    container_name_infix = "test-batch-2out"
    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command,
                    target_device=config.target_device)
    return server.start()


@pytest.fixture(scope="class")
def start_server_batch_model_auto(request):

    start_server_command_args = {"model_name": ResnetBS8.name,
                                 "model_path": ResnetBS8.model_path,
                                 "batch_size": "auto"}
    container_name_infix = "test-autobatch"
    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command,
                    target_device=config.target_device)
    return server.start()


@pytest.fixture(scope="class")
def start_server_batch_model_auto_2out(request):

    start_server_command_args = {"model_name": AgeGender.name,
                                 "model_path": AgeGender.model_path,
                                 "batch_size": "auto"}
    container_name_infix = "test-autobatch-2out"
    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command,
                    target_device=config.target_device)
    return server.start()


@pytest.fixture(scope="class")
def start_server_batch_model_bs4(request):

    start_server_command_args = {"model_name": ResnetBS8.name,
                                 "model_path": ResnetBS8.model_path,
                                 "batch_size": 4}
    container_name_infix = "test-batch4"
    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command,
                    target_device=config.target_device)
    return server.start()


@pytest.fixture(scope="class")
def start_server_batch_model_auto_bs4_2out(request):

    start_server_command_args = {"model_name": AgeGender.name,
                                 "model_path": AgeGender.model_path,
                                 "batch_size": 4}
    container_name_infix = "test-batch4-2out"
    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command)
    return server.start()
