import pytest
import shutil
from utils.model_management import wait_endpoint_setup


@pytest.fixture(scope="class")
def start_server_model_ver_policy(request, get_image, get_test_dir,
                                  get_docker_context):
    shutil.copyfile('tests/functional/model_version_policy_config.json',
                    get_test_dir +
                    '/saved_models/model_ver_policy_config.json')

    shutil.copyfile('tests/functional/mapping_config.json',
                    get_test_dir + '/saved_models/model_ver/3/'
                                   'mapping_config.json')

    client = get_docker_context
    volumes_dict = {'{}'.format(get_test_dir + '/saved_models/'):
                    {'bind': '/opt/ml', 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving config " \
              "--config_path /opt/ml/model_ver_policy_config.json " \
              "--port 9006 --rest_port 5560"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-policy',
                                      ports={'9006/tcp': 9006,
                                             '5560/tcp': 5560},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container
