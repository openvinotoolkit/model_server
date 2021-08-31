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
import os
import shutil
from pathlib import Path

import config
from object_model.ovms_binary import OvmsBinary
from object_model.ovms_docker import OvmsDocker
import logging

logger = logging.getLogger(__name__)


class Server:
    running_instances = []

    def __init__(self, request, command_args, container_name_infix, start_container_command,
                 env_vars=None, image=config.image, container_log_line=config.container_log_line,
                 server_log_level=config.log_level, target_device=None, used_models=[]):
        self.request = request
        self.command_args = command_args
        self.container_name_infix = container_name_infix
        self.start_container_command = start_container_command
        self.env_vars = env_vars
        self.image = image
        self.container_log_line = container_log_line
        self.server_log_level = server_log_level
        self.target_device = target_device
        self.started_by_fixture = request.fixturename
        self.path_to_mount = os.path.join(config.path_to_containers_dir, container_name_infix)
        self.used_models = used_models

    def _symlink_model_files_to_tmp_container_dir(self, model):
        model_dir = os.path.join(config.path_to_mount_cache, model.name)
        for root, dirs, files in os.walk(model_dir):
            rpath = os.path.relpath(root, model_dir)
            os.makedirs(os.path.join(self.path_to_mount, model.name, rpath), exist_ok=True)
            for f in files:
                src = os.path.join(root, f)
                dst = os.path.join(self.path_to_mount, model.name, rpath, f)
                if not os.path.exists(dst):
                    os.link(src, dst)

    def _prepare_directories(self):
        Path(self.path_to_mount).mkdir(parents=True, exist_ok=True)
        for f in ["config.json", "model_version_policy_config.json"]:
            shutil.copyfile(os.path.join(config.path_to_mount, f),
                            os.path.join(self.path_to_mount, f))
        for model in self.used_models:
            self._symlink_model_files_to_tmp_container_dir(model)

    def start(self):
        assert self not in Server.running_instances
        self._prepare_directories()

        if config.ovms_binary_path is not None:
            self.ovms = OvmsBinary(self.request, self.command_args, self.start_container_command, self.env_vars)
        else:
            self.ovms = OvmsDocker(self.request, self.command_args, self.container_name_infix,
                                   self.start_container_command, self.env_vars, self.image,
                                   self.container_log_line, self.server_log_level, self.target_device, server=self)
        start_result = None
        try:
            start_result = self.ovms.start()
        finally:
            if start_result is None:
                self.stop()
            else:
                Server.running_instances.append(self)
        return start_result

    def stop(self):
        self.ovms.stop()
        if self in Server.running_instances:
            Server.running_instances.remove(self)

    @classmethod
    def stop_all_instances(cls):
        for instance in cls.running_instances:
            instance.stop()

    @classmethod
    def stop_by_fixture_name(cls, fixture_name):
        instance = list(filter(lambda x: x.started_by_fixture == fixture_name, cls.running_instances))
        if instance:
            logger.debug(f"Stopping server instance spawned by {fixture_name}")
            instance[0].stop()
