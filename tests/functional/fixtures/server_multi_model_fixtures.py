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

import os
import shutil

import pytest

import config
from utils.model_management import wait_endpoint_setup
from utils.parametrization import get_tests_suffix, get_ports_for_fixture


@pytest.fixture(scope="session")
def start_server_multi_model(request, get_docker_network, start_minio_server,
                             get_minio_server_s3, get_docker_context):

    shutil.copyfile('tests/functional/config.json', config.path_to_mount + '/config.json')
    AWS_ACCESS_KEY_ID = os.getenv('MINIO_ACCESS_KEY')
    AWS_SECRET_ACCESS_KEY = os.getenv('MINIO_SECRET_KEY')
    AWS_REGION = os.getenv('AWS_REGION')

    client = get_docker_context
    network = get_docker_network

    _, ports = start_minio_server
    grpc_port, rest_port = ports["grpc_port"], ports["rest_port"]
    minio_endpoint = 'http://minio.locals3-{}.com:{}'.format(
        get_tests_suffix(), grpc_port)

    envs = ['MINIO_ACCESS_KEY=' + AWS_ACCESS_KEY_ID,
            'MINIO_SECRET_KEY=' + AWS_SECRET_ACCESS_KEY,
            'AWS_ACCESS_KEY_ID=' + AWS_ACCESS_KEY_ID,
            'AWS_SECRET_ACCESS_KEY=' + AWS_SECRET_ACCESS_KEY,
            'AWS_REGION=' + AWS_REGION,
            'S3_ENDPOINT=' + minio_endpoint,
            'https_proxy=' + os.getenv('https_proxy', ""),
            'no_proxy={}'.format(minio_endpoint)]

    volumes_dict = {'{}'.format(config.path_to_mount):
                    {'bind': '/opt/ml', 'mode': 'ro'}}

    grpc_port, rest_port = get_ports_for_fixture()

    command = "{} --config_path /opt/ml/config.json --port {} " \
              "--rest_port {} --grpc_workers 2 --rest_workers 2".\
              format(config.start_container_command, grpc_port, rest_port)

    container = client.containers.run(image=config.image, detach=True,
                                      name='ie-serving-py-test-multi-{}'.
                                      format(get_tests_suffix()),
                                      ports={'{}/tcp'.format(grpc_port):
                                             grpc_port,
                                             '{}/tcp'.format(rest_port):
                                             rest_port},
                                      remove=True, volumes=volumes_dict,
                                      environment=envs,
                                      command=command,
                                      network=network.name)

    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container, {"grpc_port": grpc_port, "rest_port": rest_port}
