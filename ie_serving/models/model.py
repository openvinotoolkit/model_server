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
from ie_serving.logger import get_logger
from abc import ABC, abstractmethod
import re

logger = get_logger(__name__)


class Model(ABC):

    def __init__(self, model_name: str, model_directory: str,
                 available_versions: list, engines: dict):
        self.model_name = model_name
        self.model_directory = model_directory
        self.versions = available_versions
        self.engines = engines
        self.default_version = max(self.versions)
        logger.info("List of available versions "
                    "for {} model: {}".format(self.model_name, self.versions))
        logger.info("Default version "
                    "for {} model is {}".format(self.model_name,
                                                self.default_version))

    @classmethod
    def build(cls, model_name: str, model_directory: str):
        logger.info("Server start loading model: {}".format(model_name))
        versions_attributes = cls.get_versions_attributes(model_directory)
        engines = cls.get_engines_for_model(versions_attributes)
        available_versions = [version_attributes['version_number'] for
                              version_attributes in versions_attributes]
        model = cls(model_name=model_name, model_directory=model_directory,
                    available_versions=available_versions, engines=engines)
        return model

    @classmethod
    def get_versions_attributes(cls, model_directory):
        versions = cls.get_versions(model_directory)
        logger.info(versions)
        versions_attributes = []
        for version in versions:
            version_number = cls.get_version_number(version=version)
            if version_number != 0:
                xml_file, bin_file, mapping_config = \
                    cls.get_version_files(version)
                if xml_file is not None and bin_file is not None:
                    version_attributes = {'xml_file': xml_file,
                                          'bin_file': bin_file,
                                          'mapping_config': mapping_config,
                                          'version_number': version_number,
                                          }
                    versions_attributes.append(version_attributes)
        return versions_attributes

    @staticmethod
    def get_version_number(version):
        version_number = re.search(r'/\d+/$', version).group(0)[1:-1]
        return int(version_number)

    @classmethod
    def get_engines_for_model(cls, versions_attributes):
        inference_engines = {}
        failures = []
        for version_attributes in versions_attributes:
            try:
                logger.info("Creating inference engine object "
                            "for version: {}".format(
                             version_attributes['version_number']))
                inference_engines[version_attributes['version_number']] = \
                    cls.get_engine_for_version(version_attributes)
            except Exception as e:
                logger.error("Error occurred while loading model "
                             "version: {}".format(version_attributes))
                logger.error("Content error: {}".format(str(e).rstrip()))
                failures.append(version_attributes)

        for failure in failures:
            versions_attributes.remove(failure)

        return inference_engines

    #   Subclass interface
    @classmethod
    @abstractmethod
    def get_versions(cls, model_directory):
        pass

    @classmethod
    @abstractmethod
    def get_version_files(cls, version):
        pass

    @classmethod
    @abstractmethod
    def _get_mapping_config(cls, version):
        pass

    @classmethod
    @abstractmethod
    def get_engine_for_version(cls, version_attributes):
        pass
