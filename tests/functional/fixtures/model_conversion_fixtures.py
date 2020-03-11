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
import requests

from pathlib import Path
from utils.model_management import convert_model


def download_file(url, dir, name):
    target_file = dir + name

    if os.path.exists(target_file):
        return target_file

    Path(dir).mkdir(parents=True, exist_ok=True)
    print("Downloading " + url + " to " + target_file + "...")
    response = requests.get(url, stream=True)
    with open(target_file, 'wb') as output:
        output.write(response.content)

    return target_file


@pytest.fixture(autouse=True, scope="session")
def resnet_multiple_batch_sizes(get_test_dir, get_docker_context):
    tensorflow_model = \
        download_file('https://download.01.org/opencv/public_models/012020/resnet-50-tf/resnet_v1-50.pb', # noqa
            get_test_dir + '/tensorflow_format/resnet/', 'resnet_v1-50.pb')

    return \
        (convert_model(get_docker_context,
                       tensorflow_model,
                       get_test_dir + '/saved_models/resnet_V1_50/1',
                       'resnet_V1_50',
                       [1, 224, 224, 3]),
         convert_model(get_docker_context,
                       tensorflow_model,
                       get_test_dir + '/saved_models/resnet_V1_50_batch4/1',
                       'resnet_V1_50_batch4',
                       [4, 224, 224, 3]),
         convert_model(get_docker_context,
                       tensorflow_model,
                       get_test_dir + '/saved_models/resnet_V1_50_batch8/1',
                       'resnet_V1_50_batch8',
                       [8, 224, 224, 3]))
