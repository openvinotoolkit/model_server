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

import config
from model.models_information import FaceDetection, PVBFaceDetectionV2, AgeGender
from utils.model_management import wait_endpoint_setup
from utils.parametrization import get_tests_suffix, get_ports_for_fixture
from utils.server import save_container_logs


@pytest.fixture(scope="class")
def start_server_model_ver_policy(request, get_docker_context):

    def finalizer():
        save_container_logs(container=container)
        container.stop()

    request.addfinalizer(finalizer)

    shutil.copyfile('tests/functional/model_version_policy_config.json',
                    config.path_to_mount + '/model_ver_policy_config.json')

    shutil.copyfile('tests/functional/mapping_config.json',
                    config.path_to_mount + '/model_ver/3/'
                                      'mapping_config.json')

    client = get_docker_context
    volumes_dict = {'{}'.format(config.path_to_mount): {'bind': '/opt/ml', 'mode': 'ro'}}

    grpc_port, rest_port = get_ports_for_fixture()

    command = "{} --config_path /opt/ml/model_ver_policy_config.json " \
              "--port {} --rest_port {}".format(config.start_container_command, grpc_port, rest_port)

    container = client.containers.run(image=config.image, detach=True,
                                      name='ie-serving-py-test-policy-{}'.
                                      format(get_tests_suffix()),
                                      ports={'{}/tcp'.format(grpc_port):
                                             grpc_port,
                                             '{}/tcp'.format(rest_port):
                                             rest_port},
                                      remove=True, volumes=volumes_dict,
                                      command=command)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container, {"grpc_port": grpc_port, "rest_port": rest_port}


@pytest.fixture(autouse=True, scope="session")
def model_version_policy_models(models_downloader):
    model_ver_dir = os.path.join(config.path_to_mount, 'model_ver')

    face_detection = os.path.join(models_downloader[FaceDetection.name], str(FaceDetection.version))
    face_detection_dir = os.path.join(model_ver_dir, '1')
    face_detection_bin = os.path.join(face_detection_dir, FaceDetection.name + ".bin")

    pvb_detection = os.path.join(models_downloader[PVBFaceDetectionV2.name], str(PVBFaceDetectionV2.version))
    pvb_detection_dir = os.path.join(model_ver_dir, '2')
    pvb_detection_bin = os.path.join(pvb_detection_dir, PVBFaceDetectionV2.name + ".bin")

    age_gender = os.path.join(models_downloader[AgeGender.name], str(AgeGender.version))
    age_gender_dir = os.path.join(model_ver_dir, '3')
    age_gender_bin = os.path.join(age_gender_dir, AgeGender.name + ".bin")

    if not (os.path.exists(model_ver_dir)
        and os.path.exists(face_detection_bin)
        and os.path.exists(pvb_detection_bin)
        and os.path.exists(age_gender_bin)):
        os.makedirs(model_ver_dir, exist_ok=True)
        copy_tree(face_detection, face_detection_dir)
        copy_tree(pvb_detection, pvb_detection_dir)
        copy_tree(age_gender, age_gender_dir)

    return face_detection_dir, pvb_detection_dir, age_gender_dir
