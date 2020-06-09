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

import docker
from retry.api import retry_call

import config
from utils.parametrization import get_ports_for_fixture
from utils.files_operation import save_container_logs_to_file
from utils.logger import get_logger

logger = get_logger(__name__)
CONTAINER_STATUS_RUNNING = "running"


class Docker:
    
    COMMON_RETRY = {"tries": 30, "delay": 2}
    GETTING_LOGS_RETRY = COMMON_RETRY
    GETTING_STATUS_RETRY = COMMON_RETRY

    def __init__(self, request, container_name, start_container_command,
                 env_vars_container=None, network="", image=config.image, container_log_line=config.container_log_line):
        self.client = docker.from_env()
        self.grpc_port, self.rest_port = get_ports_for_fixture()
        self.image = image
        self.container = None
        self.request = request
        self.container_name = container_name
        self.start_container_command = start_container_command
        self.env_vars_container = env_vars_container if env_vars_container else []
        self.network = network
        self.container_log_line = container_log_line

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

        self.ensure_container_status(status=CONTAINER_STATUS_RUNNING)
        self.ensure_logs_contains()
        return self.container, {"grpc_port": self.grpc_port, "rest_port": self.rest_port}

    def save_container_logs(self):
        if config.log_level == "DEBUG" and config.artifacts_dir != "":
            logs = self.container.logs().decode()
            save_container_logs_to_file(container=self.container, logs=logs)

    def ensure_logs(self):
        logs = str(self.container.logs())
        assert self.container_log_line in logs

    def ensure_logs_contains(self):
        return retry_call(self.ensure_logs, exceptions=AssertionError, **Docker.GETTING_LOGS_RETRY)

    def get_container_status(self):
        container = self.client.containers.get(self.container.id)
        return container.status

    def ensure_status(self, status):
        current_status = self.get_container_status()
        assert current_status == status, \
            "Not expected status for container {} found. \n " \
            "Expected: {}, \n " \
            "received: {}".format(self.container.name, status, self.container.status)

    def ensure_container_status(self, status: str = CONTAINER_STATUS_RUNNING):
        container_status = {"status": status}
        return retry_call(self.ensure_status, fkwargs=container_status,
                          exceptions=AssertionError, **Docker.GETTING_STATUS_RETRY)
