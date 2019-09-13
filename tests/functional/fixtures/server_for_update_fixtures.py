import pytest
from utils.model_management import wait_endpoint_setup


@pytest.fixture(scope="function")
def start_server_update_flow_latest(request, get_image, get_test_dir,
                                    get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet --model_path /opt/ml/update " \
              "--port 9007 --rest_port 5562"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-update-latest',
                                      ports={'9007/tcp': 9007,
                                             '5562/tcp': 5562},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="function")
def start_server_update_flow_specific(request, get_image, get_test_dir,
                                      get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = '/ie-serving-py/start_server.sh ie_serving model ' \
              '--model_name resnet --model_path /opt/ml/update ' \
              '--port 9008 --model_version_policy' \
              ' \'{"specific": { "versions":[1, 3, 4] }}\' ' \
              '--rest_port 5563'

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-'
                                           'update-specific',
                                      ports={'9008/tcp': 9008,
                                             '5563/tcp': 5563},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container
