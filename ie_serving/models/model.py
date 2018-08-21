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
import os
from ie_serving.models.ir_engine import IrEngine


class Model():

    def __init__(self, model_name: str, model_directory: str,
                 available_versions: list, engines: dict):
        self.model_name = model_name
        self.model_directory = model_directory
        self.versions = available_versions
        self.engines = engines
        self.default_version = max(self.versions)

    @classmethod
    def build(cls, model_name: str, model_directory: str):
        versions = cls.get_all_available_versions(model_directory)
        engines = cls.get_engines_for_model(versions=versions)
        available_versions = [version['version'] for version in versions]
        model = cls(model_name=model_name, model_directory=model_directory,
                    available_versions=available_versions, engines=engines)
        return model

    @staticmethod
    def get_absolute_path_to_model(specific_version_model_path):
        bin_path = glob.glob("{}/*.bin".format(specific_version_model_path))
        xml_path = glob.glob("{}/*.xml".format(specific_version_model_path))
        if xml_path[0].replace('xml', '') == bin_path[0].replace('bin', ''):
            return xml_path[0], bin_path[0]
        return None, None

    @staticmethod
    def get_model_version_number(version_path):
        folder_name = os.path.basename(os.path.normpath(version_path))
        try:
            number_version = int(folder_name)
            return number_version
        except ValueError:
            return 0

    @staticmethod
    def get_all_available_versions(model_directory):
        versions_path = glob.glob("{}/*/".format(model_directory))
        versions = []
        for version in versions_path:
            number = Model.get_model_version_number(version_path=version)
            if number != 0:
                model_xml, model_bin = Model.get_absolute_path_to_model(
                    os.path.join(model_directory, version))
                if model_xml is not None and model_bin is not None:
                    model_info = {'xml_model_path': model_xml,
                                  'bin_model_path': model_bin,
                                  'version': number}
                    versions.append(model_info)
        return versions

    @staticmethod
    def get_engines_for_model(versions):
        inference_engines = {}
        for version in versions:
            inference_engines[version['version']] = IrEngine.build(
                model_bin=version['bin_model_path'],
                model_xml=version['xml_model_path'])
        return inference_engines
