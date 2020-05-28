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
from utils.server import start_ovms_container, save_container_logs


@pytest.fixture(scope="class")
def start_server_batch_model(request, get_docker_context):

    def finalizer():
        save_container_logs(container=container)
        container.stop()

    request.addfinalizer(finalizer)

    start_server_command_args = {"model_name": ResnetBS8.name,
                                 "model_path": ResnetBS8.model_path}
    container_name_infix = "test-batch"
    container, ports = start_ovms_container(get_docker_context, start_server_command_args,
                                            container_name_infix, config.start_container_command)
    return container, ports


@pytest.fixture(scope="class")
def start_server_batch_model_2out(request, get_docker_context):

    def finalizer():
        save_container_logs(container=container)
        container.stop()

    request.addfinalizer(finalizer)

    start_server_command_args = {"model_name": AgeGender.name,
                                 "model_path": AgeGender.model_path}
    container_name_infix = "test-batch-2out"
    container, ports = start_ovms_container(get_docker_context, start_server_command_args,
                                            container_name_infix, config.start_container_command)
    return container, ports


@pytest.fixture(scope="class")
def start_server_batch_model_auto(request, get_docker_context):

    def finalizer():
        save_container_logs(container=container)
        container.stop()

    request.addfinalizer(finalizer)

    start_server_command_args = {"model_name": ResnetBS8.name,
                                 "model_path": ResnetBS8.model_path,
                                 "batch_size": "auto"}
    container_name_infix = "test-autobatch"
    container, ports = start_ovms_container(get_docker_context, start_server_command_args,
                                            container_name_infix, config.start_container_command)
    return container, ports


@pytest.fixture(scope="class")
def start_server_batch_model_auto_2out(request, get_docker_context):

    def finalizer():
        save_container_logs(container=container)
        container.stop()

    request.addfinalizer(finalizer)

    start_server_command_args = {"model_name": AgeGender.name,
                                 "model_path": AgeGender.model_path,
                                 "batch_size": "auto"}
    container_name_infix = "test-autobatch-2out"
    container, ports = start_ovms_container(get_docker_context, start_server_command_args,
                                            container_name_infix, config.start_container_command)
    return container, ports


@pytest.fixture(scope="class")
def start_server_batch_model_bs4(request, get_docker_context):

    def finalizer():
        save_container_logs(container=container)
        container.stop()

    request.addfinalizer(finalizer)

    start_server_command_args = {"model_name": ResnetBS8.name,
                                 "model_path": ResnetBS8.model_path,
                                 "batch_size": 4}
    container_name_infix = "test-batch4"
    container, ports = start_ovms_container(get_docker_context, start_server_command_args,
                                            container_name_infix, config.start_container_command)
    return container, ports


@pytest.fixture(scope="class")
def start_server_batch_model_auto_bs4_2out(request, get_docker_context):

    def finalizer():
        save_container_logs(container=container)
        container.stop()

    request.addfinalizer(finalizer)

    start_server_command_args = {"model_name": AgeGender.name,
                                 "model_path": AgeGender.model_path,
                                 "batch_size": 4}
    container_name_infix = "test-batch4-2out"
    container, ports = start_ovms_container(get_docker_context, start_server_command_args,
                                            container_name_infix, config.start_container_command)
    return container, ports
