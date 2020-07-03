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
import re
import threading
from abc import ABC, abstractmethod

from jsonschema import validate
from jsonschema.exceptions import ValidationError

from ie_serving.logger import get_logger
from ie_serving.models.model_version_status import ModelVersionStatus
from ie_serving.models.models_utils import ErrorCode
from ie_serving.schemas import latest_schema, all_schema, versions_schema

logger = get_logger(__name__)


class Model(ABC):

    def __init__(self, model_name: str, model_directory: str,
                 batch_size_param, shape_param, available_versions: list,
                 engines: dict, version_policy_filter,
                 versions_statuses: dict, update_locks: dict,
                 num_ireq: int, target_device: str, plugin_config):
        self.model_name = model_name
        self.model_directory = model_directory
        self.versions = available_versions
        self.engines = engines
        self.default_version = max(self.versions, default=-1)
        self.batch_size_param = batch_size_param
        self.shape_param = shape_param
        self.num_ireq = num_ireq
        self.version_policy_filter = version_policy_filter
        self.versions_statuses = versions_statuses

        self.target_device = target_device
        self.plugin_config = plugin_config

        [self.versions_statuses[version].set_available() for version in
         self.versions if version in self.engines.keys()]

        self.update_locks = update_locks

        logger.info("List of available versions "
                    "for {} model: {}".format(self.model_name, self.versions))
        logger.info("Default version "
                    "for {} model is {}".format(self.model_name,
                                                self.default_version))

    @classmethod
    def build(cls, model_name: str, model_directory: str, batch_size_param,
              shape_param, model_version_policy: dict = None,
              num_ireq: int = 1, target_device='CPU', plugin_config=None):

        logger.info("Server start loading model: {}".format(model_name))
        version_policy_filter = cls.get_model_version_policy_filter(
            model_version_policy)

        try:
            versions_attributes, available_versions = cls.get_version_metadata(
                model_directory, batch_size_param, shape_param,
                version_policy_filter, num_ireq, target_device, plugin_config)
        except Exception as error:
            logger.error("Error occurred while getting versions "
                         "of the model {}".format(model_name))
            logger.error("Failed reading model versions from path: {} "
                         "with error {}".format(model_directory, str(error)))
            return None

        versions_attributes = [version for version in versions_attributes
                               if version['version_number']
                               in available_versions]
        versions_statuses = dict()
        for version in available_versions:
            versions_statuses[version] = ModelVersionStatus(model_name,
                                                            version)

        update_locks = {}

        engines = cls.get_engines_for_model(model_name,
                                            versions_attributes,
                                            versions_statuses,
                                            update_locks)

        available_versions = [version_attributes['version_number'] for
                              version_attributes in versions_attributes]
        available_versions.sort()

        model = cls(model_name=model_name, model_directory=model_directory,
                    available_versions=available_versions, engines=engines,
                    batch_size_param=batch_size_param,
                    shape_param=shape_param,
                    version_policy_filter=version_policy_filter,
                    versions_statuses=versions_statuses,
                    update_locks=update_locks,
                    num_ireq=num_ireq, target_device=target_device,
                    plugin_config=plugin_config)
        return model

    def update(self):
        try:
            versions_attributes, available_versions = \
                self.get_version_metadata(
                    self.model_directory,
                    self.batch_size_param, self.shape_param,
                    self.version_policy_filter, self.num_ireq,
                    self.target_device, self.plugin_config)
        except Exception as error:
            logger.error("Error occurred while getting versions "
                         "of the model {}".format(self.model_name))
            logger.error("Failed reading model versions from path: {} "
                         "with error {}".format(self.model_directory,
                                                str(error)))
            return

        if set(available_versions) == set(self.versions):
            return

        logger.info("Server will start updating model: {}".format(
            self.model_name))
        to_create, to_delete = self._mark_differences(available_versions)
        logger.debug("Server will try to add {} versions".format(to_create))
        logger.debug(
            "Server will try to delete {} versions".format(to_delete))
        new_versions_attributes = [
            attribute for attribute in versions_attributes if
            attribute['version_number'] in to_create]

        created_engines = self.get_engines_for_model(self.model_name,
                                                     new_versions_attributes,
                                                     self.versions_statuses,
                                                     self.update_locks)
        created_versions = [attributes_to_create['version_number'] for
                            attributes_to_create in new_versions_attributes]
        self.engines.update(created_engines)
        self.versions.extend(created_versions)
        self.versions = [x for x in self.versions if x not in to_delete]
        self.versions.sort()
        self.default_version = max(self.versions, default=-1)

        [self.versions_statuses[version].set_available() for version in
         created_versions]

        logger.info("List of available versions after update "
                    "for {} model: {}".format(self.model_name, self.versions))
        logger.info("Default version after update "
                    "for {} model is {}".format(self.model_name,
                                                self.default_version))
        for version in to_delete:
            process_thread = threading.Thread(
                    target=self._delete_engine,
                    args=[version, self.update_locks])

            process_thread.start()

    def _mark_differences(self, new_versions):
        to_delete = []
        to_create = []

        for version in self.versions:
            if version not in new_versions:
                to_delete.append(version)
                self.versions_statuses[version].set_unloading()

        for version in new_versions:
            if version not in self.versions:
                to_create.append(version)
                self.versions_statuses[version] = ModelVersionStatus(
                    self.model_name, version)

        return to_create, to_delete

    def _delete_engine(self, version, update_locks):
        update_locks[version].acquire()
        try:
            self.engines[version].suppress_inference()
            self.engines[version].stop_inference_service()
            del self.engines[version]
            logger.debug("Version {} of the {} model has been removed".format(
                version, self.model_name))
            self.versions_statuses[version].set_end()
        finally:
            update_locks[version].release()

    @classmethod
    def get_version_metadata(cls, model_directory, batch_size_param,
                             shape_param, version_policy_filter, num_ireq,
                             target_device, plugin_config):
        versions_attributes = cls.get_versions_attributes(model_directory,
                                                          batch_size_param,
                                                          shape_param,
                                                          num_ireq,
                                                          target_device,
                                                          plugin_config)
        available_versions = [version_attributes['version_number'] for
                              version_attributes in versions_attributes]
        available_versions.sort()
        available_versions = version_policy_filter(available_versions)
        return versions_attributes, available_versions

    @classmethod
    def get_versions_attributes(cls, model_directory, batch_size_param,
                                shape_param, num_ireq, target_device,
                                plugin_config):
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
                                          'batch_size_param': batch_size_param,
                                          'shape_param': shape_param,
                                          'num_ireq': num_ireq,
                                          'target_device': target_device,
                                          'plugin_config': plugin_config
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
    def get_engines_for_model(cls, model_name, versions_attributes,
                              versions_statuses, update_locks):
        inference_engines = {}
        failures = []
        for version_attributes in versions_attributes:
            version_number = version_attributes['version_number']
            try:
                if version_number not in update_locks:
                    update_locks[version_number] = threading.Lock()
                update_locks[version_number].acquire()
                logger.info("Creating inference engine object "
                            "for version: {}".format(version_number))

                versions_statuses[version_number].set_loading()

                inference_engines[version_number] = \
                    cls.get_engine_for_version(model_name, version_attributes)
            except Exception as e:
                logger.error("Error occurred while loading model "
                             "version: {}".format(version_attributes))
                logger.error("Content error: {}".format(str(e).rstrip()))

                versions_statuses[version_number].set_loading(
                    ErrorCode.UNKNOWN)

                failures.append(version_attributes)
            finally:
                update_locks[version_number].release()

        for failure in failures:
            versions_attributes.remove(failure)

        return inference_engines

    @classmethod
    def _get_engine_spec(cls, model_name, version_attributes):
        return {
            'model_name': model_name,
            'model_version': version_attributes['version_number'],
            'model_bin': version_attributes['bin_file'],
            'model_xml': version_attributes['xml_file'],
            'mapping_config': version_attributes['mapping_config'],
            'batch_size_param': version_attributes['batch_size_param'],
            'shape_param': version_attributes['shape_param'],
            'num_ireq': version_attributes['num_ireq'],
            'target_device': version_attributes['target_device'],
            'plugin_config': version_attributes['plugin_config']
        }

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
    def get_engine_for_version(cls, model_name, version_attributes):
        pass
