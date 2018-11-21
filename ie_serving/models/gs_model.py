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
from ie_serving.config import MAPPING_CONFIG_FILENAME
from ie_serving.logger import get_logger
from ie_serving.models.ir_engine import IrEngine
from ie_serving.models.model import Model
import os
import re
from urllib.parse import urlparse, urlunparse

from google.cloud import storage

logger = get_logger(__name__)


class GSModel(Model):

    @staticmethod
    def gs_list_content(path):
        parsed_path = urlparse(path)
        bucket_name = parsed_path.netloc
        model_directory = parsed_path.path[1:]
        gs_client = storage.Client()
        bucket = gs_client.get_bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=model_directory)
        contents_list = []
        for blob in blobs:
            contents_list.append(blob.name)
        return contents_list

    @staticmethod
    def gs_download_file(path):
        if path is None:
            return None
        parsed_path = urlparse(path)
        bucket_name = parsed_path.netloc
        file_path = parsed_path.path[1:]
        gs_client = storage.Client()
        bucket = gs_client.get_bucket(bucket_name)
        blob = bucket.blob(file_path)
        tmp_path = os.path.join('/tmp', file_path.split(os.sep)[-1])
        blob.download_to_filename(tmp_path)
        return tmp_path

    @classmethod
    def get_versions(cls, model_directory):
        if model_directory[-1] != os.sep:
            model_directory += os.sep
        parsed_model_dir = urlparse(model_directory)
        content_list = cls.gs_list_content(model_directory)
        pattern = re.compile(parsed_model_dir.path[1:-1] + r'/\d+/$')
        versions = list(filter(pattern.match, content_list))
        return [
            urlunparse((parsed_model_dir.scheme, parsed_model_dir.netloc,
                        version, parsed_model_dir.params,
                        parsed_model_dir.query, parsed_model_dir.fragment))
            for version in versions]

    @classmethod
    def get_version_files(cls, version):
        parsed_version_path = urlparse(version)
        content_list = cls.gs_list_content(version)
        xml_pattern = re.compile(
            parsed_version_path.path[1:-1] + r'/\w+\.xml$')
        bin_pattern = re.compile(
            parsed_version_path.path[1:-1] + r'/\w+\.bin$')
        xml_file = list(filter(xml_pattern.match, content_list))
        bin_file = list(filter(bin_pattern.match, content_list))
        if xml_file[0].replace('xml', '') == \
                bin_file[0].replace('bin', ''):
            xml_file[0] = urlunparse(
                (parsed_version_path.scheme, parsed_version_path.netloc,
                 xml_file[0], parsed_version_path.params,
                 parsed_version_path.query, parsed_version_path.fragment))
            bin_file[0] = urlunparse(
                (parsed_version_path.scheme, parsed_version_path.netloc,
                 bin_file[0], parsed_version_path.params,
                 parsed_version_path.query, parsed_version_path.fragment))
            mapping_config = cls._get_mapping_config(version)
            return xml_file[0], bin_file[0], mapping_config
        return None, None, None

    @classmethod
    def _get_mapping_config(cls, version):
        content_list = cls.gs_list_content(version)
        mapping_config = urlparse(version).path[1:] + MAPPING_CONFIG_FILENAME
        if mapping_config in content_list:
            return version + MAPPING_CONFIG_FILENAME
        else:
            return None

    @classmethod
    def get_engine_for_version(cls, version_attributes):
        local_xml_file, local_bin_file, local_mapping_config = \
            cls.create_local_mirror(version_attributes)
        logger.info('Downloaded files from GCS')
        engine = IrEngine.build(model_xml=local_xml_file,
                                model_bin=local_bin_file,
                                mapping_config=local_mapping_config)
        cls.delete_local_mirror([local_xml_file, local_bin_file,
                                 local_mapping_config])
        logger.info('Deleted temporary files')
        return engine

    @classmethod
    def create_local_mirror(cls, version_attributes):
        local_xml_file = cls.gs_download_file(version_attributes['xml_file'])
        local_bin_file = cls.gs_download_file(version_attributes['bin_file'])
        local_mapping_config = cls.gs_download_file(
            version_attributes['mapping_config'])
        return local_xml_file, local_bin_file, local_mapping_config

    @classmethod
    def delete_local_mirror(cls, files_paths):
        for file_path in files_paths:
            if file_path is not None:
                os.remove(file_path)
