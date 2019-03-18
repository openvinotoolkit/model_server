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
from ie_serving.logger import get_logger
from abc import ABC, abstractmethod
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import threading
import time
from ie_serving.schemas import latest_schema, all_schema, versions_schema
import re


logger = get_logger(__name__)


class Model(ABC):

    def __init__(self, model_name: str, model_directory: str, batch_size,
                 available_versions: list, engines: dict, version_policy_filter):
        self.model_name = model_name
        self.model_directory = model_directory
        self.versions = available_versions
        self.engines = engines
        self.default_version = max(self.versions, default=-1)
        self.batch_size = batch_size
        self.version_policy_filter = version_policy_filter
        logger.info("List of available versions "
                    "for {} model: {}".format(self.model_name, self.versions))
        logger.info("Default version "
                    "for {} model is {}".format(self.model_name,
                                                self.default_version))

    @classmethod
    def build(cls, model_name: str, model_directory: str, batch_size,
              model_version_policy: dict = None):
        logger.info("Server start loading model: {}".format(model_name))
        version_policy_filter = cls.get_model_version_policy_filter(
            model_version_policy)
        versions_attributes, available_versions = cls.get_version_metadata(model_directory, batch_size, version_policy_filter)
        versions_attributes = [version for version in versions_attributes
                               if version['version_number']
                               in available_versions]
        engines = cls.get_engines_for_model(versions_attributes)
        available_versions = [version_attributes['version_number'] for
                              version_attributes in versions_attributes]
        model = cls(model_name=model_name, model_directory=model_directory,
                    available_versions=available_versions, engines=engines,
                    batch_size=batch_size, version_policy_filter=version_policy_filter)
        return model

    def update(self):
        versions_attributes, available_versions = self.get_version_metadata(self.model_directory, self.batch_size, self.version_policy_filter)
        if available_versions == self.versions:
            return
        logger.info("Server start updating model: {}".format(self.model_name))
        to_create, to_delete = self._mark_differences(available_versions)
        logger.debug("Server will try to add {} versions".format(to_create))
        logger.debug("Server will try to delete {} versions".format(to_delete))
        attributes_to_create = [attribute for attribute in versions_attributes if attribute['version_number'] in to_create]
        created_engines = self.get_engines_for_model(attributes_to_create)
        created_versions = [attributes_to_create['version_number'] for
                            attributes_to_create in attributes_to_create]
        self.engines.update(created_engines)
        self.versions.extend(created_versions)
        self.versions = [x for x in self.versions if x not in to_delete]
        self.default_version = max(self.versions, default=-1)
        logger.info("List of available versions after update"
                    "for {} model: {}".format(self.model_name, self.versions))
        logger.info("Default version after update"
                    "for {} model is {}".format(self.model_name,
                                                self.default_version))
        for version in to_delete:
            process_thread = threading.Thread(target=self._delete_engine, args=[version])
            process_thread.start()

    def _mark_differences(self, new_versions):
        to_delete = [version for version in self.versions if version not in new_versions]
        to_create = [version for version in new_versions if version not in self.versions]
        return to_create, to_delete

    def _delete_engine(self, version):
        while True:
            time.sleep(1)
            if not self.engines[version].in_use:
                logger.debug("Delete version: {}".format(version))
                del self.engines[version]
                break

    @classmethod
    def get_version_metadata(cls, model_directory, batch_size, version_policy_filter):
        versions_attributes = cls.get_versions_attributes(model_directory, batch_size)
        available_versions = [version_attributes['version_number'] for
                              version_attributes in versions_attributes]
        available_versions.sort()
        available_versions = version_policy_filter(available_versions)
        return versions_attributes, available_versions

    @classmethod
    def get_versions_attributes(cls, model_directory, batch_size):
        versions = cls.get_versions(model_directory)
        logger.debug(versions)
        versions_attributes = []
        for version in versions:
            version_number = cls.get_version_number(version=version)
            if version_number >= 0:
                xml_file, bin_file, mapping_config = \
                    cls.get_version_files(version)
                if xml_file is not None and bin_file is not None:
                    version_attributes = {'xml_file': xml_file,
                                          'bin_file': bin_file,
                                          'mapping_config': mapping_config,
                                          'version_number': version_number,
                                          'batch_size': batch_size
                                          }
                    versions_attributes.append(version_attributes)
        return versions_attributes

    @staticmethod
    def get_version_number(version):
        version_finder = re.search(r'/\d+/$', version)
        if version_finder is None:
            return -1
        return int(version_finder.group(0)[1:-1])

    @staticmethod
    def get_model_version_policy_filter(model_version_policy: dict):
        if model_version_policy is None:
            return lambda versions: versions[-1:]
        if "all" in model_version_policy:
            validate(model_version_policy, all_schema)
            return lambda versions: versions[:]
        elif "specific" in model_version_policy:
            validate(model_version_policy, versions_schema)
            return lambda versions: [version for version in versions
                                     if version in
                                     model_version_policy['specific']
                                     ['versions']]
        elif "latest" in model_version_policy:
            validate(model_version_policy, latest_schema)
            latest_number = model_version_policy['latest'].get('num_versions',
                                                               1)
            return lambda versions: versions[-latest_number:]
        raise ValidationError("ModelVersionPolicy {} is not "
                              "valid.".format(model_version_policy))

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
