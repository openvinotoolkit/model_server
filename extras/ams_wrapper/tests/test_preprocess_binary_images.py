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

import sys
import pathlib

ams_root_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(os.path.join(ams_root_path, "src"))

from preprocessing import preprocess_binary_image, ImageResizeError, ImageDecodeError


IMAGES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_images')

@pytest.fixture()
def jpg_image():
    with open(os.path.join(IMAGES_DIR, '2207159142_8206ab6984.jpg'), mode='rb') as img_file:
        binary_image = img_file.read()
    return binary_image


@pytest.fixture()
def png_image():
    with open(os.path.join(IMAGES_DIR, 'Khao_Manee_cat_in_Tamra_cat_Rattanakosin.png'),
              mode='rb') as img_file:
        binary_image = img_file.read()
    return binary_image


@pytest.fixture()
def bmp_image():
    with open(os.path.join( IMAGES_DIR, 'sails.bmp'), mode='rb') as img_file:
        binary_image = img_file.read()
    return binary_image


@pytest.mark.parametrize("image", [png_image(), jpg_image(), bmp_image()])
@pytest.mark.parametrize("reverse_input_channels", [True, False])
@pytest.mark.parametrize("target_size", [None, (10, 10), (256, 256)])
@pytest.mark.parametrize("channels_first", [True, False])
@pytest.mark.parametrize("scale", [None, 1, 1/0.017])
@pytest.mark.parametrize("standardization", [True, False])
def test_preprocess_image(image, reverse_input_channels, target_size,
                          channels_first, scale, standardization):
    decoded_image = preprocess_binary_image(image,
                                            reverse_input_channels=reverse_input_channels,
                                            target_size=target_size,
                                            channels_first=channels_first,
                                            scale=scale,
                                            standardization=standardization)

    assert decoded_image is not None
    if target_size:
        if channels_first:
            assert decoded_image.shape[1] == target_size[0]
            assert decoded_image.shape[2] == target_size[1]
        else:
            assert decoded_image.shape[0] == target_size[0]
            assert decoded_image.shape[1] == target_size[1]


@pytest.mark.parametrize("target_size", [(-1, 2), (128, 0), (0,0)])
def test_preprocess_image_wrong_target_size(png_image, target_size):
    with pytest.raises(ValueError):
        preprocess_binary_image(png_image, target_size=target_size)


def test_preprocess_image_decode_error():
    with pytest.raises(ImageDecodeError):
        preprocess_binary_image(b'not an image')
  