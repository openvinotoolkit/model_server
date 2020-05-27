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
from model.models_information import ResnetBS8
from utils.model_management import wait_endpoint_setup
from utils.parametrization import get_tests_suffix, get_ports_for_fixture
from utils.server import start_ovms_container, save_container_logs


@pytest.fixture(scope="class")
def start_server_batch_model(request, get_docker_context):

    def finalizer():
        save_container_logs(container=container)
        container.stop()

    start_server_command_args = {"model_name": ResnetBS8.name,
                                 "model_path": ResnetBS8.model_path}
    container_name_infix = "test-batch"
    request.addfinalizer(finalizer)
    container, ports = start_ovms_container(get_docker_context, start_server_command_args,
                                            container_name_infix, config.start_container_command)
    return container, ports


@pytest.fixture(scope="class")
def start_server_batch_model_2out(request, get_docker_context):

    def finalizer():
        save_container_logs(container=container)
        container.stop()

    client = get_docker_context
    volumes_dict = {'{}'.format(config.path_to_mount): {'bind': '/opt/ml',
                                                        'mode': 'ro'}}

    grpc_port, rest_port = get_ports_for_fixture()

    command = "{} --model_name age_gender " \
              "--model_path /opt/ml/age_gender " \
              "--port {} --rest_port {}".format(config.start_container_command, grpc_port, rest_port)

    request.addfinalizer(finalizer)

    container = client.containers.run(image=config.image, detach=True,
                                      name='ie-serving-py-test-batch-2out-{}'.
                                      format(get_tests_suffix()),
                                      ports={'{}/tcp'.format(grpc_port):
                                             grpc_port,
                                             '{}/tcp'.format(rest_port):
                                             rest_port},
                                      remove=True, volumes=volumes_dict,
                                      command=command)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container, {"grpc_port": grpc_port, "rest_port": rest_port}


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

    client = get_docker_context
    volumes_dict = {'{}'.format(config.path_to_mount): {'bind': '/opt/ml',
                                                        'mode': 'ro'}}

    grpc_port, rest_port = get_ports_for_fixture()

    command = "{} --model_name age_gender " \
              "--model_path /opt/ml/age_gender " \
              "--port {} --batch_size auto --rest_port {}".\
              format(config.start_container_command, grpc_port, rest_port)

    container = client.containers.run(image=config.image, detach=True,
                                      name='ie-serving-py-test-autobatch-'
                                      '2out-{}'.format(get_tests_suffix()),
                                      ports={'{}/tcp'.format(grpc_port):
                                             grpc_port,
                                             '{}/tcp'.format(rest_port):
                                             rest_port},
                                      remove=True, volumes=volumes_dict,
                                      command=command)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container, {"grpc_port": grpc_port, "rest_port": rest_port}


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

    client = get_docker_context
    volumes_dict = {'{}'.format(config.path_to_mount): {'bind': '/opt/ml',
                                                        'mode': 'ro'}}

    grpc_port, rest_port = get_ports_for_fixture()

    command = "{} --model_name age_gender " \
              "--model_path /opt/ml/age_gender " \
              "--port {} --batch_size 4 --rest_port {}".\
              format(config.start_container_command, grpc_port, rest_port)

    container = client.containers.run(image=config.image, detach=True,
                                      name='ie-serving-py-test-batch4-'
                                      '2out-{}'.format(get_tests_suffix()),
                                      ports={'{}/tcp'.format(grpc_port):
                                             grpc_port,
                                             '{}/tcp'.format(rest_port):
                                             rest_port},
                                      remove=True, volumes=volumes_dict,
                                      command=command)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container, {"grpc_port": grpc_port, "rest_port": rest_port}
