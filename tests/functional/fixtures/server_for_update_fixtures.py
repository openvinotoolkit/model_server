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
import shutil
from utils.model_management import wait_endpoint_setup
from utils.parametrization import get_tests_suffix, get_ports_for_fixture


@pytest.fixture(scope="function")
def start_server_update_flow_latest(request, get_image, get_test_dir,
                                    get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models'
    update_test_dir = path_to_mount + '/update-{}/'.format(get_tests_suffix())
    # ensure model dir is empty before starting OVMS
    shutil.rmtree(update_test_dir, ignore_errors=True)

    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}

    grpc_port, rest_port = get_ports_for_fixture()

    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet --model_path /opt/ml/update-{} " \
              "--port {} --rest_port {} --grpc_workers 1 --nireq 1".\
              format(get_tests_suffix(), grpc_port, rest_port)

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-update-'
                                      'latest-{}'.
                                      format(get_tests_suffix()),
                                      ports={'{}/tcp'.format(grpc_port):
                                             grpc_port,
                                             '{}/tcp'.format(rest_port):
                                             rest_port},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container, {"grpc_port": grpc_port, "rest_port": rest_port}


@pytest.fixture(scope="function")
def start_server_update_flow_specific(request, get_image, get_test_dir,
                                      get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models'
    update_test_dir = path_to_mount + '/update-{}/'.format(get_tests_suffix())
    # ensure model dir is empty before starting OVMS
    shutil.rmtree(update_test_dir, ignore_errors=True)

    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}

    grpc_port, rest_port = get_ports_for_fixture()

    command = '/ie-serving-py/start_server.sh ie_serving model ' \
              '--model_name resnet --model_path ' \
              '/opt/ml/update-' + get_tests_suffix() +  \
              ' --port ' + str(grpc_port) + ' --model_version_policy' \
              ' \'{"specific": { "versions":[1, 3, 4] }}\' ' \
              '--rest_port ' + str(rest_port)

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-'
                                           'update-specific-{}'.
                                      format(get_tests_suffix()),
                                      ports={'{}/tcp'.format(grpc_port):
                                             grpc_port,
                                             '{}/tcp'.format(rest_port):
                                             rest_port},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container, {"grpc_port": grpc_port, "rest_port": rest_port}
