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

from model.models_information import AgeGender, PVBDetection, PVBFaceDetectionV2, FaceDetection, PVBFaceDetectionV1
from utils.logger import get_logger

logger = get_logger(__name__)

models_to_download = [AgeGender, FaceDetection, PVBDetection, PVBFaceDetectionV1, PVBFaceDetectionV2]


def download_file(model_url_base, model_name, directory, extension, model_version = None, full_path = False):
    if model_version:
        local_model_path = os.path.join(directory, model_name, model_version)
    else:
        local_model_path = os.path.join(directory, model_name)

    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)

    local_model_full_path = os.path.join(local_model_path, model_name + extension)

    if not os.path.exists(local_model_full_path):
        logger.info("Downloading {} file to directory: {}...".format(model_name + extension, local_model_path))
        response = requests.get(model_url_base + extension, stream=True)
        with open(local_model_full_path, 'wb') as output:
            output.write(response.content)

    if full_path:
        return local_model_full_path

    return os.path.join(directory, model_name)


@pytest.fixture(autouse=True, scope="session")
def models_downloader(get_test_dir):
    models_paths = {}
    for model in models_to_download:
        models_base_path = os.path.join(get_test_dir, "saved_models")

        for extension in model.download_extensions:
            models_paths[model.name] = download_file(model.url, model.name, models_base_path, extension,
                                                     str(model.version))
    return models_paths
