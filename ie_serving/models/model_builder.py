#
# Copyright (c) 2018-2019 Intel Corporation
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
from urllib.parse import urlparse

from ie_serving.models.gs_model import GSModel
from ie_serving.models.local_model import LocalModel
from ie_serving.models.s3_model import S3Model


class ModelBuilder:
    @staticmethod
    def build(model_name: str, model_directory: str,
              model_version_policy: dict, batch_size, shape, num_ireq: int,
              target_device, plugin_config):
        parsed_path = urlparse(model_directory)
        if parsed_path.scheme == '':
            return LocalModel.build(model_name, model_directory,
                                    batch_size, shape,
                                    model_version_policy, num_ireq,
                                    target_device, plugin_config)
        elif parsed_path.scheme == 'gs':
            return GSModel.build(model_name, model_directory, batch_size,
                                 shape, model_version_policy, num_ireq,
                                 target_device, plugin_config)
        elif parsed_path.scheme == 's3':
            return S3Model.build(model_name, model_directory, batch_size,
                                 shape, model_version_policy, num_ireq,
                                 target_device, plugin_config)
