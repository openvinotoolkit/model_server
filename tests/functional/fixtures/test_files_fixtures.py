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
import os
import pytest

import config


def new_file_name(file):
    return file.replace("_template", "")


@pytest.fixture(autouse=True, scope="session")
def prepare_json(request):
    files_to_prepare = ["config_template.json", "model_version_policy_config_template.json"]
    path_to_config = "tests/functional/"

    def finalizer():
        for file in files_to_prepare:
            os.remove(os.path.join(config.path_to_mount, new_file_name(file)))

    request.addfinalizer(finalizer)

    for file_to_prepare in files_to_prepare:
        with open(path_to_config + file_to_prepare, "r") as template:
            new_file_path = os.path.join(config.path_to_mount, new_file_name(file_to_prepare))
            with open(new_file_path, "w+") as config_file:
                for line in template:
                    if "{path}" in line:
                        line = line.replace("{path}", config.models_path)
                    elif "{target_device}" in line:
                        line = line.replace("{target_device}", config.target_device)

                    config_file.write(line)
