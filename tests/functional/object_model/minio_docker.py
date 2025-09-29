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
from docker import DockerClient

import tests.functional.config as config
from tests.functional.object_model.docker import Docker
from tests.functional.utils.parametrization import generate_test_object_name
from tests.functional.utils.helpers import SingletonMeta


class MinioDocker(Docker, metaclass=SingletonMeta):
    def __init__(self, request, container_name, start_container_command=config.start_minio_container_command,
                 env_vars_container=None, image=config.minio_image,
                 container_log_line=config.container_minio_log_line):
        container_name = generate_test_object_name(prefix=container_name)
        super().__init__(request, container_name, start_container_command,
                         env_vars_container, image, container_log_line)
        self.start_container_command = start_container_command.format(self.grpc_port)
        self.start_result = None

    def start(self):
        if not self.start_result:
            self.start_container_command = self.start_container_command.format(self.grpc_port)
            try:
                self.start_result = super().start()
            finally:
                if self.start_result is None:
                    self.stop()
        return self.start_result

    @staticmethod
    def get_ip(container):
        return DockerClient().containers.get(container.id).attrs["NetworkSettings"]["IPAddress"]
