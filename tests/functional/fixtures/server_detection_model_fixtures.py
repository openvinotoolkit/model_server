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
from utils.ports import get_ports_for_fixture


@pytest.fixture(scope="class")
def start_server_face_detection_model_auto_shape(request, get_image,
                                                 get_container_suffix,
                                                 get_test_dir,
                                                 get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}

    ports = get_ports_for_fixture()
    grpc_port, rest_port = ports["grpc_port"], ports["rest_port"]
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name face_detection --model_path " \
              "/opt/ml/face-detection-retail-0004 " \
              "--port {} --rest_port {} --shape auto " \
              "--grpc_workers 4 --nireq 4".format(grpc_port, rest_port)

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-auto-shape-{}'.
                                      format(get_container_suffix),
                                      ports={'{}/tcp'.format(grpc_port):
                                             grpc_port,
                                             '{}/tcp'.format(rest_port):
                                             rest_port},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container, ports


@pytest.fixture(scope="class")
def start_server_face_detection_model_named_shape(request, get_image,
                                                  get_container_suffix,
                                                  get_test_dir,
                                                  get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}

    ports = get_ports_for_fixture()
    grpc_port, rest_port = ports["grpc_port"], ports["rest_port"]

    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name face_detection --model_path " \
              "/opt/ml/face-detection-retail-0004 " \
              "--port " + grpc_port + " --rest_port " + rest_port + \
              " --shape \"{\\\"data\\\": \\\"(1, 3, 600, 600)\\\"}\" " \
              "--grpc_workers 4 --rest_workers 2 " \
              "--nireq 2"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-named-shape-{}'.
                                      format(get_container_suffix),
                                      ports={'{}/tcp'.format(grpc_port):
                                             grpc_port,
                                             '{}/tcp'.format(rest_port):
                                             rest_port},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container, ports


@pytest.fixture(scope="class")
def start_server_face_detection_model_nonamed_shape(request, get_image,
                                                    get_container_suffix,
                                                    get_test_dir,
                                                    get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    ports = get_ports_for_fixture()
    grpc_port, rest_port = ports["grpc_port"], ports["rest_port"]
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name face_detection --model_path " \
              "/opt/ml/face-detection-retail-0004 " \
              "--port {} --rest_port {} " \
              "--shape \"(1, 3, 600, 600)\" " \
              "--rest_workers 4 --nireq 2".format(grpc_port, rest_port)

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-nonamed-'
                                      'shape-{}'.format(get_container_suffix),
                                      ports={'{}/tcp'.format(grpc_port):
                                             grpc_port,
                                             '{}/tcp'.format(rest_port):
                                             rest_port},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container, ports
