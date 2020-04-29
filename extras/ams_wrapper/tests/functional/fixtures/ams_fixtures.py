import pytest



@pytest.fixture(scope='session')
def ams_object_detection_model_endpoint(ams_host, ams_port):
    return 'http://{host}:{port}/entity-object-detection'.format(host=ams_host,
                                                                 port=ams_port)


@pytest.fixture(scope='session')
def png_object_detection_image():
    with open('images/car.png', mode='rb') as image_file:
        return image_file.read()


@pytest.fixture(scope='session')
def jpg_object_detection_image():
    with open('images/car.jpg', mode='rb') as image_file:
        return image_file.read()


@pytest.fixture(scope='session')
def bmp_object_detection_image():
    with open('images/car.bmp', mode='rb') as image_file:
        return image_file.read()


@pytest.fixture(scope='session')
def small_object_detection_image():
    with open('images/car_small.png', mode='rb') as image_file:
        return image_file.read()


@pytest.fixture(scope='session')
def medium_object_detection_image():
    with open('images/car.png', mode='rb') as image_file:
        return image_file.read()


@pytest.fixture(scope='session')
def large_object_detection_image():
    with open('images/car_large.png', mode='rb') as image_file:
        return image_file.read()


@pytest.fixture(scope='session')
def object_detection_image_one_entity():
    return b'one'


@pytest.fixture(scope='session')
def object_detection_image_five_entities():
    return b'five'