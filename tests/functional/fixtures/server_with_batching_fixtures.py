import pytest
from utils.model_management import wait_endpoint_setup


@pytest.fixture(scope="class")
def start_server_batch_model(request, get_image, get_test_dir,
                             get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet --model_path /opt/ml/resnet_V1_50_batch8 " \
              "--port 9003 --rest_port 5557"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-batch',
                                      ports={'9003/tcp': 9003,
                                             '5557/tcp': 5557},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_batch_model_auto(request, get_image, get_test_dir,
                                  get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet --model_path /opt/ml/resnet_V1_50_batch8 " \
              "--port 9005 --batch_size auto --rest_port 5559"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-autobatch',
                                      ports={'9005/tcp': 9005,
                                             '5559/tcp': 5559},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_batch_model_bs4(request, get_image, get_test_dir,
                                 get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet --model_path /opt/ml/resnet_V1_50_batch8 " \
              "--port 9004 --batch_size 4 --rest_port 5558"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-batch4',
                                      ports={'9004/tcp': 9004,
                                             '5558/tcp': 5558},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container
