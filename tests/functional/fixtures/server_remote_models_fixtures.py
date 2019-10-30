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
from utils.model_management import wait_endpoint_setup


@pytest.fixture(scope="class")
def start_server_single_model_from_gc(request, get_image, get_test_dir,
                                      get_docker_context):
    GOOGLE_APPLICATION_CREDENTIALS = \
        os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    client = get_docker_context
    envs = ['GOOGLE_APPLICATION_CREDENTIALS=/etc/gcp.json']
    volumes_dict = {GOOGLE_APPLICATION_CREDENTIALS: {'bind': '/etc/gcp.json',
                                                     'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet " \
              "--model_path gs://inference-eu/ml-test " \
              "--port 9000 --target_device CPU --nireq 4 --plugin_config " \
              "\"{\\\"CPU_THROUGHPUT_STREAMS\\\": \\\"2\\\", " \
              "\\\"CPU_THREADS_NUM\\\": \\\"4\\\"}\""

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-single-gs',
                                      ports={'9000/tcp': 9000},
                                      remove=True, volumes=volumes_dict,
                                      environment=envs,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_single_model_from_s3(request, get_image, get_test_dir,
                                      get_docker_context):
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION')

    client = get_docker_context
    envs = ['AWS_ACCESS_KEY_ID=' + AWS_ACCESS_KEY_ID,
            'AWS_SECRET_ACCESS_KEY=' + AWS_SECRET_ACCESS_KEY,
            'AWS_REGION=' + AWS_REGION]
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet " \
              "--model_path s3://inference-test-aipg/resnet_v1_50 " \
              "--port 9000"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-single-s3',
                                      ports={'9000/tcp': 9000},
                                      remove=True,
                                      environment=envs,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container
