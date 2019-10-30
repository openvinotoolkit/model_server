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
from utils.model_management import wait_endpoint_setup


@pytest.fixture(scope="session")
def start_server_multi_model(request, get_image, get_test_dir,
                             get_docker_context):
    shutil.copyfile('tests/functional/config.json',
                    get_test_dir + '/saved_models/config.json')

    GOOGLE_APPLICATION_CREDENTIALS = \
        os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION')

    client = get_docker_context
    envs = ['GOOGLE_APPLICATION_CREDENTIALS=/etc/gcp.json',
            'AWS_ACCESS_KEY_ID=' + AWS_ACCESS_KEY_ID,
            'AWS_SECRET_ACCESS_KEY=' + AWS_SECRET_ACCESS_KEY,
            'AWS_REGION=' + AWS_REGION]
    volumes_dict = {'{}'.format(get_test_dir + '/saved_models/'):
                    {'bind': '/opt/ml', 'mode': 'ro'},
                    GOOGLE_APPLICATION_CREDENTIALS:
                        {'bind': '/etc/gcp.json', 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving config " \
              "--config_path /opt/ml/config.json --port 9001 " \
              "--rest_port 5561 --grpc_workers 2 --rest_workers 2"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-multi',
                                      ports={'9001/tcp': 9001,
                                             '5561/tcp': 5561},
                                      remove=True, volumes=volumes_dict,
                                      environment=envs,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container
