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
from object_model.ovms_binary import OvmsBinary
from object_model.ovms_docker import OvmsDocker
from utils.logger import get_logger

logger = get_logger(__name__)


class Server:
    def __init__(self, request, command_args, container_name_infix, start_container_command,
                 env_vars=None, image=config.image, container_log_line=config.container_log_line,
                 server_log_level=config.log_level, target_device=None):
        self.request = request
        self.command_args = command_args
        self.container_name_infix = container_name_infix
        self.start_container_command = start_container_command
        self.env_vars = env_vars
        self.image = image
        self.container_log_line = container_log_line
        self.server_log_level = server_log_level
        self.target_device = target_device

    def start(self):
        if config.ovms_binary_path is not None:
            ovms = OvmsBinary(self.request, self.command_args, self.start_container_command, self.env_vars)
        else:
            ovms = OvmsDocker(self.request, self.command_args, self.container_name_infix,
                              self.start_container_command, self.env_vars,
                              self.image, self.container_log_line, self.server_log_level, self.target_device)
        return ovms.start()
