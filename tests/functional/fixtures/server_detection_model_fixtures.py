import pytest
from utils.model_management import wait_endpoint_setup


@pytest.fixture(scope="class")
def start_server_face_detection_model(request, get_image, get_test_dir,
                                      get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir + '/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name face_detection --model_path " \
              "/opt/ml/face-detection-retail-0004 " \
              "--port 9000 --rest_port 5555 --shape auto"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-face-detection',
                                      ports={'9000/tcp': 9000,
                                             '5555/tcp': 5555},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container
