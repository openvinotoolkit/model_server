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

import pytest
from utils.model_management import wait_endpoint_setup


@pytest.fixture(scope="class")
def start_server_single_model(request, get_image, get_test_dir,
                              get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet --model_path /opt/ml/resnet_V1_50 " \
              "--port 9000 --rest_port 5555 --plugin_config " \
              "\"{\\\"CPU_THROUGHPUT_STREAMS\\\": " \
              "\\\"CPU_THROUGHPUT_AUTO\\\"}\""

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-single',
                                      ports={'9000/tcp': 9000,
                                             '5555/tcp': 5555},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_with_mapping(request, get_image, get_test_dir,
                              get_docker_context):
    shutil.copyfile('tests/functional/mapping_config.json',
                    get_test_dir + '/saved_models/resnet_2_out/1/'
                                   'mapping_config.json')
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet_2_out --model_path /opt/ml/resnet_2_out " \
              "--port 9002 --rest_port 5556"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-2-out',
                                      ports={'9002/tcp': 9002,
                                             '5556/tcp': 5556},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container
