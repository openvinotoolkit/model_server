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
def start_server_face_detection_model_auto_shape(request, get_image,
                                                 get_test_dir,
                                                 get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name face_detection --model_path " \
              "/opt/ml/face-detection-retail-0004 " \
              "--port 9010 --rest_port 5565 --shape auto " \
              "--grpc_workers 4 --nireq 4"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-auto-shape',
                                      ports={'9010/tcp': 9010,
                                             '5565/tcp': 5565},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_face_detection_model_named_shape(request, get_image,
                                                  get_test_dir,
                                                  get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name face_detection --model_path " \
              "/opt/ml/face-detection-retail-0004 " \
              "--port 9011 --rest_port 5566 " \
              "--shape \"{\\\"data\\\": \\\"(1, 3, 600, 600)\\\"}\" " \
              "--grpc_workers 4 --rest_workers 2 --nireq 2"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-named-shape',
                                      ports={'9011/tcp': 9011,
                                             '5566/tcp': 5566},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_face_detection_model_nonamed_shape(request, get_image,
                                                    get_test_dir,
                                                    get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name face_detection --model_path " \
              "/opt/ml/face-detection-retail-0004 " \
              "--port 9012 --rest_port 5567 " \
              "--shape \"(1, 3, 600, 600)\" " \
              "--rest_workers 4 --nireq 2"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-nonamed-shape',
                                      ports={'9012/tcp': 9012,
                                             '5567/tcp': 5567},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container
