import pytest
import os
from utils.model_management import wait_endpoint_setup


@pytest.fixture(scope="class")
def start_server_single_model_from_gc(request, get_image, get_test_dir,
                                      get_docker_context):
    GOOGLE_APPLICATION_CREDENTIALS = \
        os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    client = get_docker_context
    envs = ['GOOGLE_APPLICATION_CREDENTIALS=/etc/gcp.json']
    volumes_dict = {GOOGLE_APPLICATION_CREDENTIALS: {'bind': '/etc/gcp.json',
                                                     'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet " \
              "--model_path gs://inference-eu/ml-test " \
              "--port 9000"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-single-gs',
                                      ports={'9000/tcp': 9000},
                                      remove=True, volumes=volumes_dict,
                                      environment=envs,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_single_model_from_s3(request, get_image, get_test_dir,
                                      get_docker_context):
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION')

    client = get_docker_context
    envs = ['AWS_ACCESS_KEY_ID=' + AWS_ACCESS_KEY_ID,
            'AWS_SECRET_ACCESS_KEY=' + AWS_SECRET_ACCESS_KEY,
            'AWS_REGION=' + AWS_REGION]
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet " \
              "--model_path s3://inference-test-aipg/resnet_v1_50 " \
              "--port 9000"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-single-s3',
                                      ports={'9000/tcp': 9000},
                                      remove=True,
                                      environment=envs,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container
