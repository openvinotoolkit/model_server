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
import requests


def download_model(model_name, model_folder, model_version_folder, dir):
    model_url_base = "https://storage.googleapis.com/inference-eu/models_zoo/"\
                     + model_name + "/frozen_" + model_name

    if not os.path.exists(dir + model_folder + model_version_folder):
        print("Downloading " + model_name + " model...")
        print(dir)
        os.makedirs(dir + model_folder + model_version_folder)
        response = requests.get(model_url_base + '.bin', stream=True)
        with open(
                dir + model_folder + model_version_folder + model_name +
                '.bin', 'wb') as output:
            output.write(response.content)
        response = requests.get(model_url_base + '.xml', stream=True)
        with open(
                dir + model_folder + model_version_folder + model_name +
                '.xml', 'wb') as output:
            output.write(response.content)
    return dir + model_folder + model_version_folder + model_name + '.bin', \
        dir + model_folder + model_version_folder + model_name + '.xml'


@pytest.fixture(autouse=True, scope="session")
def face_detection_model_downloader(get_test_dir):
    return download_model('face-detection-retail-0004',
                          'face-detection-retail-0004/',
                          '1/',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True, scope="session")
def resnet_v1_50_model_downloader(get_test_dir):
    return download_model('resnet_V1_50', 'resnet_V1_50/', '1/',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True, scope="session")
def pnasnet_large_model_downloader(get_test_dir):
    return download_model('pnasnet_large', 'pnasnet_large/', '1/',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True, scope="session")
def resnet_2_out_model_downloader(get_test_dir):
    return download_model('resnet_2_out', 'resnet_2_out/', '1/',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True, scope="session")
def resnet_8_batch_model_downloader(get_test_dir):
    return download_model('resnet_V1_50_batch8', 'resnet_V1_50_batch8/', '1/',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True, scope="session")
def download_two_models(get_test_dir):
    model1_info = download_model('resnet_V1_50', 'resnet_V1_50/', '1/',
                                 get_test_dir + '/saved_models/')
    model2_info = download_model('pnasnet_large', 'pnasnet_large/', '1/',
                                 get_test_dir + '/saved_models/')
    return [model1_info, model2_info]


@pytest.fixture(autouse=True, scope="session")
def download_two_model_versions(get_test_dir):
    model1_info = download_model('resnet_V1_50', 'resnet/', '1/',
                                 get_test_dir + '/saved_models/')
    model2_info = download_model('resnet_V2_50', 'resnet/', '2/',
                                 get_test_dir + '/saved_models/')
    return [model1_info, model2_info]
