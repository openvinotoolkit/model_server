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
from utils.model_management import wait_endpoint_setup


@pytest.fixture(scope="class")
def start_server_batch_model(request, get_image, get_test_dir,
                             get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet --model_path /opt/ml/resnet_V1_50_batch8 " \
              "--port 9003 --rest_port 5557"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-batch',
                                      ports={'9003/tcp': 9003,
                                             '5557/tcp': 5557},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_batch_model_2out(request, get_image, get_test_dir,
                                  get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name age_gender " \
              "--model_path /opt/ml/age-gender-recognition-retail-0013 " \
              "--port 9006 --rest_port 5560"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-batch-2out',
                                      ports={'9006/tcp': 9006,
                                             '5560/tcp': 5560},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_batch_model_auto(request, get_image, get_test_dir,
                                  get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet --model_path /opt/ml/resnet_V1_50_batch8 " \
              "--port 9005 --batch_size auto --rest_port 5559"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-autobatch',
                                      ports={'9005/tcp': 9005,
                                             '5559/tcp': 5559},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_batch_model_auto_2out(request, get_image, get_test_dir,
                                       get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name age_gender " \
              "--model_path /opt/ml/age-gender-recognition-retail-0013 " \
              "--port 9007 --batch_size auto --rest_port 5561"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-autobatch-2out',
                                      ports={'9007/tcp': 9007,
                                             '5561/tcp': 5561},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_batch_model_bs4(request, get_image, get_test_dir,
                                 get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet --model_path /opt/ml/resnet_V1_50_batch8 " \
              "--port 9004 --batch_size 4 --rest_port 5558"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-batch4',
                                      ports={'9004/tcp': 9004,
                                             '5558/tcp': 5558},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_batch_model_auto_bs4_2out(request, get_image, get_test_dir,
                                           get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name age_gender " \
              "--model_path /opt/ml/age-gender-recognition-retail-0013 " \
              "--port 9008 --batch_size 4 --rest_port 5562"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-batch4-2out',
                                      ports={'9008/tcp': 9008,
                                             '5562/tcp': 5562},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container
