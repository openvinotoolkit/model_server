#
# Copyright (c) 2019-2020 Intel Corporation
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
import config
from command_wrappers.server import start_ovms_container_command
from utils.model_management import wait_endpoint_setup
from utils.parametrization import get_ports_for_fixture, get_tests_suffix
from utils.files_operation import save_container_logs_to_file


def start_ovms_container(client, command_args, container_name_infix, start_container_command, env_vars_container=None):
    if env_vars_container is None:
        env_vars_container = []
    container_name_prefix = config.image.split(":")[0].split("/")[-1]
    volumes_dict = {'{}'.format(config.path_to_mount): {'bind': '/opt/ml',
                                                        'mode': 'ro'}}

    grpc_port, rest_port = get_ports_for_fixture()
    command_args["port"] = grpc_port
    command_args["rest_port"] = rest_port
    command = start_ovms_container_command(start_container_command, command_args)
    container_name = "{}-{}-{}".format(container_name_prefix, container_name_infix, get_tests_suffix())

    container = client.containers.run(image=config.image, detach=True,
                                      name=container_name,
                                      ports={'{}/tcp'.format(grpc_port):
                                             grpc_port,
                                             '{}/tcp'.format(rest_port):
                                             rest_port},
                                      remove=True, volumes=volumes_dict,
                                      command=command, environment=env_vars_container)
    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"
    return container, {"grpc_port": grpc_port, "rest_port": rest_port}


def save_container_logs(container):
    if config.log_level == "DEBUG" and config.artifacts_dir != "":
        logs = container.logs().decode()
        save_container_logs_to_file(container=container, logs=logs)
