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
from ie_serving.config import StorageType
from ie_serving.logger import get_logger
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

    @classmethod
    def get_versions_path(cls, model_directory):
        if model_directory[-1] != os.sep:
            model_directory += os.sep
        parsed_model_dir = urlparse(model_directory)
        content_list = cls.gs_list_content(model_directory)
        pattern = re.compile(parsed_model_dir.path[1:-1] + '/\d+/$')
        version_dirs = list(filter(pattern.match, content_list))
        return [
            urlunparse((parsed_model_dir.scheme, parsed_model_dir.netloc,
                        version_dir, parsed_model_dir.params,
                        parsed_model_dir.query, parsed_model_dir.fragment))
            for version_dir in version_dirs]

    @classmethod
    def get_full_path_to_model(cls, specific_version_model_path):
            parsed_version_path = urlparse(specific_version_model_path)
            content_list = cls.gs_list_content(specific_version_model_path)
            xml_pattern = re.compile(
                parsed_version_path.path[1:-1] + '/\w+\.xml$')
            bin_pattern = re.compile(
                parsed_version_path.path[1:-1] + '/\w+\.bin$')
            xml_path = list(filter(xml_pattern.match, content_list))
            bin_path = list(filter(bin_pattern.match, content_list))
            if xml_path[0].replace('xml', '') == \
                    bin_path[0].replace('bin', ''):
                xml_path[0] = urlunparse(
                    (parsed_version_path.scheme, parsed_version_path.netloc,
                     xml_path[0], parsed_version_path.params,
                     parsed_version_path.query, parsed_version_path.fragment))
                bin_path[0] = urlunparse(
                    (parsed_version_path.scheme, parsed_version_path.netloc,
                     bin_path[0], parsed_version_path.params,
                     parsed_version_path.query, parsed_version_path.fragment))
                return StorageType.GS, xml_path[0], bin_path[0]
            return None, None, None
