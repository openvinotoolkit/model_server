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
import pytest

import tests.functional.config as config
from tests.functional.object_model.minio_docker import MinioDocker
from tests.functional.object_model.server import Server


@pytest.fixture(scope="session")
def start_server_multi_model(request, start_minio_server, get_minio_server_s3):

    aws_access_key_id = os.getenv('MINIO_ACCESS_KEY')
    aws_secret_access_key = os.getenv('MINIO_SECRET_KEY')
    aws_region = os.getenv('AWS_REGION')

    minio_container, ports = start_minio_server
    grpc_port, rest_port = ports["grpc_port"], ports["rest_port"]

    if config.ovms_binary_path:
        minio_endpoint = "http://localhost:{}".format(grpc_port)
    else:
        minio_endpoint = "{}:{}".format(MinioDocker.get_ip(minio_container), grpc_port)

    envs = ['MINIO_ACCESS_KEY=' + aws_access_key_id,
            'MINIO_SECRET_KEY=' + aws_secret_access_key,
            'AWS_ACCESS_KEY_ID=' + aws_access_key_id,
            'AWS_SECRET_ACCESS_KEY=' + aws_secret_access_key,
            'AWS_REGION=' + aws_region,
            'S3_ENDPOINT=' + minio_endpoint]

    start_server_command_args = {"config_path": "{}/config.json".format(config.models_path),
                                 "grpc_workers": 2,
                                 "rest_workers": 2}
    container_name_infix = "test-multi"
    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command, envs)
    return server.start()
