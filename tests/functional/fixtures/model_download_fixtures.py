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

import tests.functional.config as config
from tests.functional.model.models_information import AgeGender, PVBDetection, PVBFaceDetectionV2, FaceDetection, PVBFaceDetectionV1, ResnetONNX
from tests.functional.utils.model_management import download_missing_file
import logging

logger = logging.getLogger(__name__)

models_to_download = [AgeGender, FaceDetection, PVBDetection, PVBFaceDetectionV1, PVBFaceDetectionV2, ResnetONNX]


def download_file(model_url_base, model_name, directory, extension, model_version = None, full_path = False):
    if model_version:
        local_model_path = os.path.join(directory, model_name, model_version)
    else:
        local_model_path = os.path.join(directory, model_name)

    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)

    local_model_full_path = os.path.join(local_model_path, model_name + extension)

    download_missing_file(model_url_base + extension, local_model_full_path)

    if full_path:
        return local_model_full_path

    return os.path.join(directory, model_name)


@pytest.fixture(autouse=True, scope="session")
def models_downloader():
    models_paths = {}
    for model in models_to_download:
        for extension in model.download_extensions:
            models_paths[model.name] = download_file(model.url, model.name, config.path_to_mount_cache, extension,
                                                     str(model.version))
    return models_paths


@pytest.fixture(autouse=True, scope="session")
def copy_cached_models_to_test_dir(models_downloader):
    cached_models_paths = models_downloader
    os.makedirs(config.path_to_mount, exist_ok=True)

    for model_name, cached_model_dir in cached_models_paths.items():
        dest_model_dir = os.path.join(config.path_to_mount, model_name)

        logger.info("Copying model {} from cache to {}".format(model_name, dest_model_dir))
        if not os.path.exists(dest_model_dir):
            shutil.copytree(cached_model_dir, dest_model_dir)
