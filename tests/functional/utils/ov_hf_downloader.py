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

import os
from datetime import datetime, timezone
from huggingface_hub import HfApi, snapshot_download

from tests.functional.config import huggingface_token
from tests.functional.utils.logger import get_logger
from tests.functional.utils.test_framework import get_dir_latest_mtime, remove_dir_tree, swap_directory

logger = get_logger(__name__)


class OVHfDownloader:

    def __init__(self, model_type, model_base_path=None):
        if not huggingface_token:
            raise Exception(
                "Provide huggingfacace_token with TT_HUGGINGFACE_TOKEN or TT_HUGGINGFACE_TOKEN_FILE_PATH envs"
            )
        self.api = HfApi(token=huggingface_token)
        self.model = model_type()
        self.model_name = self.model.name
        if model_base_path is None:
            self.model_local_path = self.model.model_path_on_host
        else:
            self.model_local_path = os.path.join(model_base_path, self.model.name)

    def check_and_update_hf_model(self):
        repo_info = self.api.repo_info(self.model_name)
        local_latest_mtime = get_dir_latest_mtime(self.model_local_path)
        local_latest_dt = datetime.fromtimestamp(local_latest_mtime, tz=timezone.utc) if local_latest_mtime else None

        if local_latest_dt and repo_info.last_modified <= local_latest_dt:
            print(f"No files to update for model: {self.model_name}")
            return

        print(f"Download OVHf model: {self.model_name}")
        staging_path = self.model_local_path + "_staging"
        if os.path.exists(staging_path):
            remove_dir_tree(staging_path)
        self.download_model(model_dir=staging_path)
        swap_directory(self.model_local_path, staging_path)

    def download_model(self, model_name=None, model_dir=None, force_download=False):
        snapshot_download(
            repo_id=self.model_name if model_name is None else model_name,
            local_dir=self.model_local_path if model_dir is None else model_dir,
            token=huggingface_token,
            force_download=force_download,
        )
