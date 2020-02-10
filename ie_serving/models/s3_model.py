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
import os
import re
from urllib.parse import urlparse, urlunparse

import boto3
from botocore import UNSIGNED
from botocore import exceptions
from botocore.client import Config

from ie_serving.config import GLOBAL_CONFIG, S3_CONFIG
from ie_serving.logger import get_logger
from ie_serving.models.ir_engine import IrEngine
from ie_serving.models.model import Model

logger = get_logger(__name__)


class S3Model(Model):
    @classmethod
    def s3_list_content(cls, path):
        s3_resource = boto3.resource(
            's3', endpoint_url=S3_CONFIG['endpoint'],
            aws_access_key_id=S3_CONFIG['access_key_id'],
            aws_secret_access_key=S3_CONFIG['secret_access_key'],
            config=Config(signature_version=S3_CONFIG['signature']),
            region_name=S3_CONFIG['region'])
        parsed_path = urlparse(path)
        my_bucket = s3_resource.Bucket(parsed_path.netloc)
        content_list = []
        for object in my_bucket.objects.filter(Prefix=parsed_path.path[1:]):
            content_list.append(object.key)
        return content_list

    @classmethod
    def s3_download_file(cls, path):
        if path is None:
            return None
        parsed_path = urlparse(path)
        bucket_name = parsed_path.netloc
        file_path = parsed_path.path[1:]
        tmp_path = os.path.join('/tmp', file_path.split(os.sep)[-1])
        try:
            s3_client = boto3.client(
                's3', endpoint_url=S3_CONFIG['endpoint'],
                aws_access_key_id=S3_CONFIG['access_key_id'],
                aws_secret_access_key=S3_CONFIG['secret_access_key'],
                config=Config(signature_version=S3_CONFIG['signature']),
                region_name=S3_CONFIG['region'])
            s3_transfer = boto3.s3.transfer.S3Transfer(s3_client)
            s3_transfer.download_file(bucket_name, file_path, tmp_path)
        except exceptions.ClientError:
            s3_client = boto3.client(
                's3', endpoint_url=S3_CONFIG['endpoint'],
                aws_access_key_id=S3_CONFIG['access_key_id'],
                aws_secret_access_key=S3_CONFIG['secret_access_key'],
                config=Config(signature_version=UNSIGNED),
                region_name=S3_CONFIG['region'])
            s3_transfer = boto3.s3.transfer.S3Transfer(s3_client)
            s3_transfer.download_file(bucket_name, file_path, tmp_path)
        return tmp_path

    @classmethod
    def get_versions(cls, model_directory):
        if model_directory[-1] != os.sep:
            model_directory += os.sep
        parsed_model_dir = urlparse(model_directory)
        content_list = cls.s3_list_content(model_directory)
        pattern = re.compile('(' + parsed_model_dir.path[1:-1] + r'/\d+/).*$')
        versions = set([m.group(1) for m in (pattern.match(element) for
                                             element in content_list) if m])

        return [urlunparse((parsed_model_dir.scheme, parsed_model_dir.netloc,
                            version, parsed_model_dir.params,
                            parsed_model_dir.query, parsed_model_dir.fragment))
                for version in versions]

    @classmethod
    def get_version_files(cls, version):
        parsed_version_path = urlparse(version)
        content_list = cls.s3_list_content(version)
        xml_pattern = re.compile(
            parsed_version_path.path[1:-1] + r'/[^/]+\.xml$')
        bin_pattern = re.compile(
            parsed_version_path.path[1:-1] + r'/[^/]+\.bin$')
        xml_file = list(filter(xml_pattern.match, content_list))
        bin_file = list(filter(bin_pattern.match, content_list))
        if len(xml_file) != 0 and len(bin_file) != 0:
            if os.path.splitext(xml_file[0])[0] == \
                    os.path.splitext(bin_file[0])[0]:
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
        content_list = cls.s3_list_content(version)
        mapping_config = urlparse(version).path[1:] + GLOBAL_CONFIG[
            'mapping_config_filename']
        if mapping_config in content_list:
            return version + GLOBAL_CONFIG['mapping_config_filename']
        else:
            return None

    @classmethod
    def get_engine_for_version(cls, model_name, version_attributes):
        version_attributes['xml_file'], version_attributes['bin_file'], \
            version_attributes['mapping_config'] = cls.create_local_mirror(
            version_attributes)
        logger.info('Downloaded files from S3')

        engine_spec = cls._get_engine_spec(model_name, version_attributes)
        engine = IrEngine.build(**engine_spec)

        cls.delete_local_mirror([version_attributes['xml_file'],
                                 version_attributes['bin_file'],
                                 version_attributes['mapping_config']])
        logger.info('Deleted temporary files')
        return engine

    @classmethod
    def create_local_mirror(cls, version_attributes):
        local_xml_file = cls.s3_download_file(version_attributes['xml_file'])
        local_bin_file = cls.s3_download_file(version_attributes['bin_file'])
        local_mapping_config = cls.s3_download_file(
            version_attributes['mapping_config'])
        return local_xml_file, local_bin_file, local_mapping_config

    @classmethod
    def delete_local_mirror(cls, files_paths):
        for file_path in files_paths:
            if file_path is not None:
                os.remove(file_path)
