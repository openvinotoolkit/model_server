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

from ie_serving.config import GLOBAL_CONFIG
from ie_serving.logger import get_logger
from ie_serving.models.ir_engine import IrEngine
from ie_serving.models.model import Model
import os

logger = get_logger(__name__)


class LocalModel(Model):

    @classmethod
    def get_versions(cls, model_directory):
        if model_directory[-1] != os.sep:
            model_directory += os.sep
        return glob.glob("{}/*/".format(model_directory))

    @classmethod
    def get_version_files(cls, version):
        bin_file = glob.glob("{}*.bin".format(version))
        xml_file = glob.glob("{}*.xml".format(version))
        if len(xml_file) != 0 and len(bin_file) != 0:
            if os.path.splitext(xml_file[0])[0] == \
                    os.path.splitext(bin_file[0])[0]:
                mapping_config = cls._get_mapping_config(version)
                return xml_file[0], bin_file[0], mapping_config
        return None, None, None

    @classmethod
    def _get_mapping_config(cls, version):
        mapping_config = glob.glob(version + GLOBAL_CONFIG[
            'mapping_config_filename'])
        if len(mapping_config) == 1:
            return mapping_config[0]
        return None

    @classmethod
    def get_engine_for_version(cls, model_name, version_attributes):
        engine_spec = cls._get_engine_spec(model_name, version_attributes)
        engine = IrEngine.build(**engine_spec)
        return engine
