#
# Copyright (c) 2020 Intel Corporation
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
from tests.functional.fixtures.model_download_fixtures import download_file
from tests.functional.model.models_information import Resnet, ResnetBS4, ResnetBS8
import logging
from tests.functional.utils.model_management import convert_model

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True, scope="session")
def resnet_multiple_batch_sizes(get_docker_context):
    resnet_to_convert = [Resnet, ResnetBS4, ResnetBS8]
    converted_models = []
    tensorflow_model_path = download_file(model_url_base=Resnet.url, model_name=Resnet.name,
                                          directory=os.path.join(config.test_dir_cache, Resnet.local_conversion_dir),
                                          extension=Resnet.download_extensions[0],
                                          full_path=True)

    for resnet in resnet_to_convert:
        logger.info("Converting model {}".format(resnet.name))
        input_shape = list(resnet.input_shape)

        converted_model = convert_model(get_docker_context, tensorflow_model_path,
                                        config.path_to_mount_cache + '/{}/{}'.format(resnet.name, resnet.version),
                                        resnet.name, input_shape)
        converted_models.append(converted_model)

    return converted_models


@pytest.fixture(autouse=True, scope="session")
def copy_cached_resnet_models(resnet_multiple_batch_sizes):
    cached_resnet_models = resnet_multiple_batch_sizes

    for cached_model_path_bin, _ in cached_resnet_models:
        cached_model_dir = os.path.dirname(cached_model_path_bin)
        dest_model_dir = cached_model_dir.replace(config.path_to_mount_cache, config.path_to_mount)

        logger.info("Copying resnet model from cache to {}".format(dest_model_dir))
        if not os.path.exists(dest_model_dir):
            shutil.copytree(cached_model_dir, dest_model_dir)
