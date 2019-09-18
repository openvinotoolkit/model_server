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
from distutils.dir_util import copy_tree

import pytest
from utils.model_management import wait_endpoint_setup


@pytest.fixture(scope="class")
def start_server_model_ver_policy(request, get_image, get_test_dir,
                                  get_docker_context):
    shutil.copyfile('tests/functional/model_version_policy_config.json',
                    get_test_dir +
                    '/saved_models/model_ver_policy_config.json')

    shutil.copyfile('tests/functional/mapping_config.json',
                    get_test_dir + '/saved_models/model_ver/3/'
                                   'mapping_config.json')

    client = get_docker_context
    volumes_dict = {'{}'.format(get_test_dir + '/saved_models/'):
                    {'bind': '/opt/ml', 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving config " \
              "--config_path /opt/ml/model_ver_policy_config.json " \
              "--port 9006 --rest_port 5560"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-policy',
                                      ports={'9006/tcp': 9006,
                                             '5560/tcp': 5560},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(autouse=True, scope="session")
def model_version_policy_models(get_test_dir,
                                download_two_model_versions,
                                resnet_2_out_model_downloader):
    model_ver_dir = os.path.join(get_test_dir, 'saved_models', 'model_ver')
    resnets = download_two_model_versions
    resnet_1 = os.path.dirname(resnets[0][0])
    resnet_1_dir = os.path.join(model_ver_dir, '1')
    resnet_2 = os.path.dirname(resnets[1][0])
    resnet_2_dir = os.path.join(model_ver_dir, '2')
    resnet_2_out = os.path.dirname(resnet_2_out_model_downloader[0])
    resnet_2_out_dir = os.path.join(model_ver_dir, '3')
    if not os.path.exists(model_ver_dir):
        os.makedirs(model_ver_dir)
        copy_tree(resnet_1, resnet_1_dir)
        copy_tree(resnet_2, resnet_2_dir)
        copy_tree(resnet_2_out, resnet_2_out_dir)

    return resnet_1_dir, resnet_2_dir, resnet_2_out_dir
