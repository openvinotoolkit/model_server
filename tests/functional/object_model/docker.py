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
from grp import getgrnam
from os import getuid
import os
import time
from typing import List
from datetime import datetime

import docker
from docker.types import DeviceRequest
import logging
from retry.api import retry_call
from tests.functional.utils.files_operation import get_path_friendly_test_name

import tests.functional.config as config
from tests.functional.utils.grpc import port_manager_grpc
from tests.functional.utils.rest import port_manager_rest
from tests.functional.constants.constants import TARGET_DEVICE_HDDL, TARGET_DEVICE_GPU, TARGET_DEVICE_CUDA, TARGET_DEVICE_CPU, TARGET_DEVICE_MYRIAD


logger = logging.getLogger(__name__)
CONTAINER_STATUS_RUNNING = "running"
TERMINAL_STATUSES = ["exited"]

TARGET_DEVICE_CONFIGURATION = {
    TARGET_DEVICE_CPU: {
        'volumes': {},
        "privileged": False,
    },

    TARGET_DEVICE_GPU: {
        'volumes': {},
        "devices": ["/dev/dri:/dev/dri:mrw"],
        "privileged": False,
        "user": None,
        "group_add": [getgrnam('render').gr_gid, getgrnam('video').gr_gid]
    },

    TARGET_DEVICE_CUDA: {
        'volumes': {},
        "privileged": False,
        'device_requests':  [DeviceRequest(count=-1, capabilities=[['gpu']])]
    },

    TARGET_DEVICE_MYRIAD: {
        'volumes': {"/dev/bus/usb": {'bind': "/dev/bus/usb", 'mode': 'ro'}},
        'privileged': False,
        "user": f"{getuid()}:{getgrnam('users').gr_gid}",
        "device_cgroup_rules": ["c 189:* rmw"]
    },

    TARGET_DEVICE_HDDL: {
        "volumes": {"/var/tmp": {"bind": "/var/tmp", "mode": "rw"}},
        "devices": ["/dev/ion:/dev/ion:mrw"],
        "privileged": False,
        "user": "root"
    },
}


class Docker:
    
    COMMON_RETRY = {"tries": 360, "delay": 0.5}
    GETTING_LOGS_RETRY = COMMON_RETRY
    GETTING_STATUS_RETRY = COMMON_RETRY

    def __init__(self, request, container_name, start_container_command,
                 env_vars_container=None, image=config.image, container_log_line=config.container_log_line,
                 server=None):
        self.server = server
        self.client = docker.from_env()
        self.grpc_port = port_manager_grpc.get_port()
        self.rest_port = port_manager_rest.get_port()
        self.image = image
        self.container = None
        self.request = request
        self.container_name = container_name
        self.start_container_command = start_container_command
        self.env_vars_container = env_vars_container if env_vars_container else []
        self.container_log_line = container_log_line
        self.logs = ""

    def start(self):
        start_result = None
        try:
            start_result = self._start()
        finally:
            if start_result is None:
                self.stop()  # Failed to start container so clean it up
        return start_result

    def _start(self):
        logger.info(f"Starting container: {self.container_name}")

        ports = {'{}/tcp'.format(self.grpc_port): self.grpc_port, '{}/tcp'.format(self.rest_port): self.rest_port}
        device_cfg = TARGET_DEVICE_CONFIGURATION[config.target_device]
        volumes_dict = {config.path_to_mount: {'bind': '/opt/ml', 'mode': 'ro'}}
        device_cfg['volumes'].update(volumes_dict)

        self.container = self.client.containers.run(image=self.image, detach=True,
                                                    name=self.container_name,
                                                    ports=ports,
                                                    command=self.start_container_command,
                                                    environment=self.env_vars_container,
                                                    **device_cfg)
        self.ensure_container_status(status=CONTAINER_STATUS_RUNNING, terminal_statuses=TERMINAL_STATUSES)
        self.ensure_logs_contains()
        logger.info(f"Container started grpc_port:{self.grpc_port}\trest_port:{self.rest_port}")
        logger.debug(f"Container starting command args: {self.start_container_command}")
        return self.container, {"grpc_port": self.grpc_port, "rest_port": self.rest_port}

    def stop(self):
        if self.container is not None:
            logger.info(f"Stopping container: {self.container_name}")
            self.container.stop(timeout=10)
            self.save_container_logs()
            self.container.remove(v=True)
            self.container = None
            port_manager_grpc.release_port(self.grpc_port)
            port_manager_rest.release_port(self.rest_port)
            logger.info(f"Container successfully closed and removed: {self.container_name}")

    def save_container_logs(self):
        logs = self.get_logs()
        if config.log_level == "DEBUG":
            logger.info(logs)
        if config.artifacts_dir != "":
            location = getattr(self.request.node, "location", None)
            self.save_container_logs_to_file(logs=logs, location=location)

    def get_logs(self):
        self.logs = self.container.logs().decode()
        return self.logs

    def ensure_logs(self):
        logs = self.get_logs()
        for log_line in self.container_log_line:
            if log_line not in logs:
                assert False, f"Not found required phrase {log_line}"
  

    def ensure_logs_contains(self):
        result = None
        try:
            result = retry_call(self.ensure_logs, exceptions=AssertionError, **Docker.GETTING_LOGS_RETRY)
        except:
            if config.log_level == "DEBUG":
                logger.info(str(self.get_logs()))
        return result

    def get_container_status(self):
        container = self.client.containers.get(self.container.id)
        return container.status

    def ensure_status(self, status, terminal_statuses=None):
        current_status = self.get_container_status()
        logger.debug(f"Ensure container status, expected_status={status}\t current_status={current_status}")
        ovms_logs = self.container.logs().decode('ascii')
        if terminal_statuses is not None and current_status in terminal_statuses:
            raise RuntimeError("Received terminal status '{}' for container {} - \nOVMS LOGS:\n===\n{}\n===".format(
                current_status, self.container_name, ovms_logs))
        assert current_status == status, \
            "Not expected status for container {} found. \n" \
            "Expected: {}, \n" \
            "received: {}\n" \
            "Ovms logs: {}\n".format(self.container.name, status, self.container.status, ovms_logs)

    def ensure_container_status(self, status: str = CONTAINER_STATUS_RUNNING,
                                terminal_statuses: List[str] = None):
        container_statuses = {"status": status}
        if terminal_statuses:
            container_statuses["terminal_statuses"] = terminal_statuses
        time.sleep(1)
        return retry_call(self.ensure_status, fkwargs=container_statuses,
                          exceptions=AssertionError, **Docker.GETTING_STATUS_RETRY)

    def save_container_logs_to_file(self, logs, dir_path: str = config.artifacts_dir, location=None):
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if location:
            file_name = f"ovms_{get_path_friendly_test_name(location)}_{time_stamp}.log"
        else:
            file_name = f"ovms_{self.server.started_by_fixture.lstrip('start_')}_{time_stamp}.log"
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, "w+") as text_file:
            text_file.write(logs)
