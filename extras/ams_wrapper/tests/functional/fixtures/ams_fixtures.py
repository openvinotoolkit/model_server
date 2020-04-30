import os

import pytest


IMAGES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_images')


@pytest.fixture(scope='session')
def ams_object_detection_model_endpoint(ams_host, ams_port):
    return 'http://{host}:{port}/vehicle-detection'.format(host=ams_host,
                                                          port=ams_port)


@pytest.fixture(scope='session')
def png_object_detection_image() -> bytes:
    with open(os.path.join(IMAGES_DIR, 'single_car_small.png'), mode='rb') as image_file:
        return image_file.read()


@pytest.fixture(scope='session')
def jpg_object_detection_image() -> bytes:
    with open(os.path.join(IMAGES_DIR, 'single_car_small.jpg'), mode='rb') as image_file:
        return image_file.read()


@pytest.fixture(scope='session')
def bmp_object_detection_image() -> bytes:
    with open(os.path.join(IMAGES_DIR, 'single_car_small.bmp'), mode='rb') as image_file:
        return image_file.read()


@pytest.fixture(scope='session')
def small_object_detection_image() -> bytes:
   with open(os.path.join(IMAGES_DIR, 'single_car_small.jpg'), mode='rb') as image_file:
        return image_file.read()


@pytest.fixture(scope='session')
def medium_object_detection_image() -> bytes:
    with open(os.path.join(IMAGES_DIR, 'single_car_medium.jpg'), mode='rb') as image_file:
        return image_file.read()


@pytest.fixture(scope='session')
def large_object_detection_image() -> bytes:
    with open(os.path.join(IMAGES_DIR, 'single_car_large.png'), mode='rb') as image_file:
        return image_file.read()


@pytest.fixture(scope='session')
def object_detection_image_no_entities() -> bytes:
    with open(os.path.join(IMAGES_DIR, 'white.png'), mode='rb') as image_file:
        return image_file.read()


@pytest.fixture(scope='session')
def object_detection_image_one_entity() -> bytes:
    with open(os.path.join(IMAGES_DIR, 'single_car_medium.jpg'), mode='rb') as image_file:
        return image_file.read()


@pytest.fixture(scope='session')
def object_detection_image_two_entities() -> bytes:
    with open(os.path.join(IMAGES_DIR, 'two_cars.jpg'), mode='rb') as image_file:
        return image_file.read()
