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

import numpy as np
import pytest
import requests


def input_data_downloader(numpy_url, get_test_dir):
    filename = numpy_url.split("/")[-1]
    if not os.path.exists(get_test_dir + '/' + filename):
        response = requests.get(numpy_url, stream=True)
        with open(get_test_dir + '/' + filename, 'wb') as output:
            output.write(response.content)
    imgs = np.load(get_test_dir + '/' + filename, mmap_mode='r',
                   allow_pickle=False)
    imgs = imgs.transpose((0, 3, 1, 2))  # transpose to adjust from NHWC>NCHW
    print(imgs.shape)
    return imgs


@pytest.fixture(autouse=True, scope="session")
def input_data_downloader_v1_224(get_test_dir):
    return input_data_downloader(
        'https://storage.googleapis.com/inference-eu/models_zoo/resnet_V1_50/datasets/10_v1_imgs.npy', # noqa
        get_test_dir)


@pytest.fixture(autouse=True, scope="session")
def input_data_downloader_v3_331(get_test_dir):
    return input_data_downloader(
        'https://storage.googleapis.com/inference-eu/models_zoo/pnasnet_large/datasets/10_331_v3_imgs.npy', # noqa
        get_test_dir)
