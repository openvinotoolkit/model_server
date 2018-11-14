#
# Copyright (c) 2018 Intel Corporation
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
import glob

from ie_serving.config import MAPPING_CONFIG_FILENAME
from ie_serving.logger import get_logger
from ie_serving.models.model import Model
import os

logger = get_logger(__name__)


class LocalModel(Model):

    @classmethod
    def get_versions_path(cls, model_directory):
        if model_directory[-1] != os.sep:
            model_directory += os.sep
        return glob.glob("{}/*/".format(model_directory))

    @classmethod
    def get_full_path_to_model(cls, specific_version_model_path):
        bin_path = glob.glob("{}*.bin".format(specific_version_model_path))
        xml_path = glob.glob("{}*.xml".format(specific_version_model_path))
        if xml_path[0].replace('xml', '') == bin_path[0].replace('bin', ''):
            mapping_config_path = cls._get_path_to_mapping_config(
                specific_version_model_path)
            return xml_path[0], bin_path[0], mapping_config_path
        return None, None, None

    @classmethod
    def _get_path_to_mapping_config(cls, specific_version_model_path):

        config_path = glob.glob(specific_version_model_path +
                                MAPPING_CONFIG_FILENAME)
        if len(config_path) == 1:
            return config_path[0]
        return None
