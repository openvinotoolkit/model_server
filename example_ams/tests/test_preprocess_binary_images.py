#
# Copyright (c) 2018 Intel Corporation
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


import pytest

from ams.preprocess_binary_images import preprocess_binary_image

@pytest.fixture()
def jpg_image():
    with open('test_images/2207159142_8206ab6984.jpg', mode='rb') as img_file:
        binary_image = img_file.read()
    return binary_image


@pytest.fixture()
def png_image():
    with open('test_images/Khao_Manee_cat_in_Tamra_cat_Rattanakosin.png', mode='rb') as img_file:
        binary_image = img_file.read()
    return binary_image


@pytest.fixture()
def bmp_image():
    with open('test_images/sails.bmp', mode='rb') as img_file:
        binary_image = img_file.read()
    return binary_image

@pytest.mark.parametrize("reverse_input_channels", [True, False])
@pytest.mark.parametrize("scale", [None, 1, 1/255, 1/0.017])
@pytest.mark.parametrize("standardization", [True, False])
def test_preprocess_jpg(jpg_image, reverse_input_channels, scale, standardization):
    assert preprocess_binary_image(jpg_image, reverse_input_channels=reverse_input_channels,
     scale=scale, standardization=standardization) is not None


@pytest.mark.parametrize("reverse_input_channels", [True, False])
@pytest.mark.parametrize("scale", [None, 1, 1/255, 1/0.017])
@pytest.mark.parametrize("standardization", [True, False])
def test_preprocess_png(png_image, reverse_input_channels, scale, standardization):
    assert preprocess_binary_image(png_image, reverse_input_channels=reverse_input_channels,
     scale=scale, standardization=standardization) is not None


@pytest.mark.parametrize("reverse_input_channels", [True, False])
@pytest.mark.parametrize("scale", [None, 1, 1/255, 1/0.017])
@pytest.mark.parametrize("standardization", [True, False])
def test_preprocess_bmp(bmp_image, reverse_input_channels, scale, standardization):
    assert preprocess_binary_image(bmp_image, reverse_input_channels=reverse_input_channels,
     scale=scale, standardization=standardization) is not None
