import pytest



@pytest.fixture(scope='session')
def ams_object_detection_model_endpoint(ams_host, ams_port):
    return 'http://{host}:{port}/entity-object-detection'.format(host=ams_host,
                                                                 port=ams_port)


@pytest.fixture(scope='session')
def png_object_detection_image():
    with open('images/005405_001.png', mode='rb') as png_file:
        return png_file.read()


@pytest.fixture(scope='session')
def jpg_object_detection_image():
    return b'jpg'


@pytest.fixture(scope='session')
def bmp_object_detection_image():
    return b'bmp'


@pytest.fixture(scope='session')
def small_object_detection_image():
    return b'small'


@pytest.fixture(scope='session')
def medium_object_detection_image():
    return b'medium'


@pytest.fixture(scope='session')
def large_object_detection_image():
    return b'large'


@pytest.fixture(scope='session')
def object_detection_image_one_entity():
    return b'one'


@pytest.fixture(scope='session')
def object_detection_image_five_entities():
    return b'five'