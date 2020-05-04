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
from typing import Tuple

import pytest


IMAGES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_images')


# TODO: use actual ams instance, started in docker container, instead of mock server
# TODO: for now, we assume that mock server is running on localhost:8000
@pytest.fixture(scope='session')
def ams_object_detection_model_endpoint() -> str:
    return 'http://{host}:{port}/vehicle-detection'.format(host='localhost',
                                                           port=8000)


@pytest.fixture(scope='session')
def png_object_detection_image() -> str:
    return os.path.join(IMAGES_DIR, 'single_car_small.png')


@pytest.fixture(scope='session')
def jpg_object_detection_image() -> str:
    return os.path.join(IMAGES_DIR, 'single_car_small.jpg')


@pytest.fixture(scope='session')
def bmp_object_detection_image() -> str:
    return os.path.join(IMAGES_DIR, 'single_car_small.bmp')


@pytest.fixture(scope='session')
def small_object_detection_image() -> str:
   return os.path.join(IMAGES_DIR, 'single_car_small.jpg')


@pytest.fixture(scope='session')
def medium_object_detection_image() -> str:
    return os.path.join(IMAGES_DIR, 'single_car_medium.jpg')


@pytest.fixture(scope='session')
def large_object_detection_image() -> str:
    return os.path.join(IMAGES_DIR, 'single_car_large.png')


@pytest.fixture(scope='session')
def object_detection_image_no_entities() -> str:
    return os.path.join(IMAGES_DIR, 'white.png')


@pytest.fixture(scope='session')
def object_detection_image_one_entity() -> str:
    return os.path.join(IMAGES_DIR, 'single_car_medium.jpg')


@pytest.fixture(scope='session')
def object_detection_image_two_entities() -> str:
    return os.path.join(IMAGES_DIR, 'two_cars.jpg')
