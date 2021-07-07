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

import config
from command_wrappers.server import start_ovms_container_command
from object_model.docker import Docker
from utils.parametrization import generate_test_object_name


class OvmsDocker(Docker):
    def __init__(self, request, command_args, container_name_infix, start_container_command,
                 env_vars=None, image=config.image, container_log_line=config.container_log_line,
                 server_log_level=config.log_level, target_device=None):
        self.command_args = command_args
        self.container_name_infix = container_name_infix
        self.server_log_level = server_log_level
        container_name_prefix = image.split(":")[0].split("/")[-1]
        container_name = generate_test_object_name(prefix="{}-{}".format(container_name_prefix, container_name_infix))
        super().__init__(request, container_name, start_container_command,
                         env_vars, image, container_log_line)
        self.command_args["port"] = self.grpc_port
        self.command_args["rest_port"] = self.rest_port
        self.command_args["log_level"] = self.server_log_level
        if target_device:
            self.command_args["target_device"] = target_device
        self.start_container_command = start_ovms_container_command(self.start_container_command, self.command_args)

    def start(self):
        return super().start()
