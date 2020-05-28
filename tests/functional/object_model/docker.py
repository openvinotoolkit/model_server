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

import time

import docker

import config
from utils.model_management import serving_condition
from utils.parametrization import get_ports_for_fixture
from utils.files_operation import save_container_logs_to_file
from utils.logger import get_logger

logger = get_logger(__name__)


class Docker:

    def __init__(self, request, container_name, start_container_command,
                 env_vars_container=None, network="", image=config.image):
        self.client = docker.from_env()
        self.grpc_port, self.rest_port = get_ports_for_fixture()
        self.image = image
        self.container = None
        self.request = request
        self.container_name = container_name
        self.start_container_command = start_container_command
        self.env_vars_container = env_vars_container if env_vars_container else []
        self.network = network

    def start(self):

        def finalizer():
            self.save_container_logs()
            self.container.stop()

        self.request.addfinalizer(finalizer)

        volumes_dict = {'{}'.format(config.path_to_mount): {'bind': '/opt/ml',
                                                            'mode': 'ro'}}

        self.container = self.client.containers.run(image=self.image, detach=True,
                                                    name=self.container_name,
                                                    ports={'{}/tcp'.format(self.grpc_port):
                                                               self.grpc_port,
                                                           '{}/tcp'.format(self.rest_port):
                                                               self.rest_port},
                                                    remove=True, volumes=volumes_dict,
                                                    command=self.start_container_command,
                                                    environment=self.env_vars_container,
                                                    network=self.network)
        running = self.wait_endpoint_setup()
        assert running is True, "docker container was not started successfully"
        return self.container, {"grpc_port": self.grpc_port, "rest_port": self.rest_port}

    def save_container_logs(self):
        if config.log_level == "DEBUG" and config.artifacts_dir != "":
            logs = self.container.logs().decode()
            save_container_logs_to_file(container=self.container, logs=logs)

    def wait_endpoint_setup(self, condition=serving_condition, timeout=60,
                            container_log_line=config.container_log_line):
        start_time = time.time()
        tick = start_time
        running = False
        while tick - start_time < timeout:
            tick = time.time()
            try:
                if condition(self.container, container_log_line):
                    running = True
                    break
            except Exception as e:
                time.sleep(1)
        logger.debug("Logs from container:")
        logger.debug("\n".join(str(self.container.logs()).split("\\n")))
        #  extra delay to ensure docker endpoint is ready
        time.sleep(2)
        return running
