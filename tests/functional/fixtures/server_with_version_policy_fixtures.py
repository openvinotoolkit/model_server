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
import shutil
from distutils.dir_util import copy_tree

import tests.functional.config as config
from tests.functional.model.models_information import FaceDetection, PVBFaceDetectionV2, AgeGender
from tests.functional.object_model.server import Server


@pytest.fixture(scope="session")
def start_server_model_ver_policy(request):

    shutil.copyfile('tests/functional/mapping_config.json',
                    config.path_to_mount + '/model_ver/3/mapping_config.json')

    start_server_command_args = {"config_path": "{}/model_version_policy_config.json".format(config.models_path)}
    container_name_infix = "test-batch4-2out"

    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command)
    return server.start()


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
