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
import tarfile

face_detection_model_url = "https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/face-detection-retail-0004/FP32/face-detection-retail-0004"  # noqa
age_gender_recognition_model_url = "https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013"  # noqa
pvb_detection_model_url = "https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/person-vehicle-bike-detection-crossroad-0078/FP32/person-vehicle-bike-detection-crossroad-0078"  # noqa
vehicle_detection_adas_bin_0001_url = "https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/1/vehicle-detection-adas-binary-0001/FP32-INT1/vehicle-detection-adas-binary-0001"  # noqa
vehicle_data_url = "https://download.01.org/opencv/openvino_training_extensions/datasets/crossroad/crossroad_85.tar.gz"  # noqa
vehicle_attributes_url = "https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/1/vehicle-attributes-recognition-barrier-0039/FP32/vehicle-attributes-recognition-barrier-0039"  # noqa
vehicle_attributes_data_url = "https://download.01.org/opencv/openvino_training_extensions/datasets/vehicle_attributes/cars_100.tar.gz"  # noqa


def download_model(model_url_base, model_name, model_version, dir):
    local_model_path = os.path.join(dir, model_name, model_version)
    local_bin_path = os.path.join(local_model_path,
                                  "{}.{}".format(model_name, "bin"))
    local_xml_path = os.path.join(local_model_path,
                                  "{}.{}".format(model_name, "xml"))
    if not os.path.exists(local_model_path):
        print("Downloading " + model_name + " model...")
        print(dir)
        os.makedirs(local_model_path)
        response = requests.get(model_url_base + '.bin', stream=True)
        with open(local_bin_path, 'wb') as output:
            output.write(response.content)
        response = requests.get(model_url_base + '.xml', stream=True)
        with open(local_xml_path, 'wb') as output:
            output.write(response.content)
    return local_bin_path, local_xml_path


def download_data(data_url_base, model_name, model_version, dir):
    local_data_path = os.path.join(dir, model_name, model_version, "data")

    local_tar_path = os.path.join(local_data_path,
                                  "{}.{}".format(model_name, "tar.gz"))
    if not os.path.exists(local_data_path):
        print("Downloading " + model_name + " data...")
        print(dir)
        os.makedirs(local_data_path)
        response = requests.get(data_url_base, stream=True)
        with open(local_tar_path, 'wb') as output:
            output.write(response.content)

    tar_file = tarfile.open(local_tar_path)
    tar_file.extractall(local_data_path)

    return local_data_path


@pytest.fixture(autouse=True, scope="session")
def face_detection_model_downloader(get_test_dir):
    return download_model(face_detection_model_url,
                          'face-detection-retail-0004',
                          '1',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True, scope="session")
def age_gender_model_downloader(get_test_dir):
    return download_model(age_gender_recognition_model_url,
                          'age-gender-recognition-retail-0013', '1',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True, scope="session")
def pvb_model_downloader(get_test_dir):
    return download_model(pvb_detection_model_url,
                          'person-vehicle-bike-detection-crossroad-0078', '1',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True, scope="session")
def vehicle_adas_model_downloader(get_test_dir):
    return download_model(vehicle_detection_adas_bin_0001_url,
                          'vehicle-detection-adas-binary-0001', '1',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True, scope="session")
def vehicle_adas_data_downloader(get_test_dir):
    return download_data(vehicle_data_url,
                         'vehicle-detection-adas-binary-0001', '1',
                         get_test_dir + '/saved_data/')


@pytest.fixture(autouse=True, scope="session")
def vehicle_attributes_model_downloader(get_test_dir):
    return download_model(vehicle_attributes_url,
                          'vehicle-attributes-recognition-barrier-0039', '1',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True, scope="session")
def vehicle_attributes_data_downloader(get_test_dir):
    return download_data(vehicle_attributes_data_url,
                         'vehicle-attributes-recognition-barrier-0039', '1',
                         get_test_dir + '/saved_data/')


"""
Use converted models instead

@pytest.fixture(autouse=True, scope="session")
def download_two_models(get_test_dir):
    model1_info = download_model(resnet_model_url, 'resnet50-binary-0001', '1',
                                 get_test_dir + '/saved_models/')
    model2_info = download_model(face_detection_model_url,
                                 'face-detection-retail-0004', '1',
                                 get_test_dir + '/saved_models/')
    return [model1_info, model2_info]
"""


@pytest.fixture(autouse=True, scope="session")
def download_two_model_versions(get_test_dir):
    model1_info = download_model(face_detection_model_url,
                                 'pvb_face_multi_version', '1',
                                 get_test_dir + '/saved_models/')
    model2_info = download_model(pvb_detection_model_url,
                                 'pvb_face_multi_version', '2',
                                 get_test_dir + '/saved_models/')
    return [model1_info, model2_info]
