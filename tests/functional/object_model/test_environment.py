#
# Copyright (c) 2026 Intel Corporation
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

import json
import os
import socket
from pathlib import Path

from tests.functional.utils.logger import get_logger
from tests.functional.config import server_address

logger = get_logger(__name__)


class TestEnvironment(object):
    __test__ = False
    current = None

    def __init__(self, base_dir):
        self.base_dir = base_dir

    @staticmethod
    def update_model_files(model, models_dir):
        if hasattr(model, "max_position_embeddings") and model.max_position_embeddings is not None:
            config_file_path = os.path.join(models_dir[0], model.name, "config.json")
            if os.path.exists(config_file_path):
                with open(config_file_path, "r") as fo:
                    config_data = json.load(fo)
                config_data["max_position_embeddings"] = model.max_position_embeddings
                with open(config_file_path, "w") as fo:
                    json.dump(config_data, fo)
                logger.info(
                    f"max_position_embeddings value was updated to {model.max_position_embeddings} "
                    f"in model's config file: {config_file_path}."
                )

    def prepare_container_folders(self, dir_name, models):
        """
        Method execute prepare_resources on each model.

        Parameters:
        name (str): Tmp name of a container
        models (List[ModelInfo]): List of resources.

        Returns:
        str: location of resources directory path on host (container folder)
        Set(str): set of models directory path on host
        """
        resources_dir = os.path.join(self.base_dir, dir_name)
        Path(resources_dir).mkdir(parents=True, exist_ok=True)
        models_dir_on_host = set()
        for model in models or []:
            models_dir = model.prepare_resources(resources_dir)
            models_dir_on_host.update(models_dir)
            self.update_model_files(model, models_dir)
        return resources_dir, models_dir_on_host

    @staticmethod
    def get_server_address():
        final_server_address = (
            os.environ.get("REMOTE_SERVER_ADDRESS")
            if os.environ.get("REMOTE_SERVER_ADDRESS") is not None
            else server_address
        )
        return final_server_address

    @staticmethod
    def get_local_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.connect(("<broadcast>", 0))
        return s.getsockname()[0]

    @staticmethod
    def get_ip_from_hostname(hostname):
        ip = socket.gethostbyname(hostname)
        return ip
