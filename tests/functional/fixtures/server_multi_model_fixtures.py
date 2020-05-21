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
from utils.parametrization import get_tests_suffix
from utils.server import start_ovms_container
from utils.server import save_container_logs


@pytest.fixture(scope="session")
def start_server_multi_model(request, get_docker_network, start_minio_server,
                             get_minio_server_s3, get_docker_context):

    def finalizer():
        save_container_logs(container=container)
        container.stop()

    request.addfinalizer(finalizer)

    shutil.copyfile('tests/functional/config.json', config.path_to_mount + '/config.json')
    aws_access_key_id = os.getenv('MINIO_ACCESS_KEY')
    aws_secret_access_key = os.getenv('MINIO_SECRET_KEY')
    aws_region = os.getenv('AWS_REGION')

    network = get_docker_network

    _, ports = start_minio_server
    grpc_port, rest_port = ports["grpc_port"], ports["rest_port"]
    minio_endpoint = 'http://minio.locals3-{}.com:{}'.format(
        get_tests_suffix(), grpc_port)

    envs = ['MINIO_ACCESS_KEY=' + aws_access_key_id,
            'MINIO_SECRET_KEY=' + aws_secret_access_key,
            'AWS_ACCESS_KEY_ID=' + aws_access_key_id,
            'AWS_SECRET_ACCESS_KEY=' + aws_secret_access_key,
            'AWS_REGION=' + aws_region,
            'S3_ENDPOINT=' + minio_endpoint,
            'https_proxy=' + os.getenv('https_proxy', ""),
            'no_proxy={}'.format(minio_endpoint)]

    start_server_command_args = {"config_path": "/opt/ml/config.json",
                                 "grpc_workers": 2,
                                 "rest_workers": 2}
    container_name_infix = "test-multi"
    container, ports = start_ovms_container(get_docker_context, start_server_command_args,
                                            container_name_infix, config.start_container_command, envs, network.name)

    return container, ports
