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
import pytest

from ie_serving.models.s3_model import S3Model


@pytest.mark.parametrize("content_list, versions", [
    (['model/3/', 'model/3.txt', 'model/sub/2/', 'model/one'],
     ['s3://bucket/model/3/']),
    (['model/3/model.xml', 'model/3.txt', 'model/sub/2/', 'model/one'],
     ['s3://bucket/model/3/']),
    (['model/3/something.xml', 'model/3/something.bin', 'model/2/4/3/file',
      'model/1/'],
     ['s3://bucket/model/3/', 's3://bucket/model/2/', 's3://bucket/model/1/']),
    (['model/dir1/', 'model/3.txt', 'model/dir2/2/', 'model/one/file'], []),
])
def test_get_versions(mocker, content_list, versions):
    list_content_mocker = mocker.patch('ie_serving.models.s3_model.S3Model.'
                                       's3_list_content')
    list_content_mocker.return_value = content_list

    output = S3Model.get_versions('s3://bucket/model')
    assert set(output) == set(versions)


@pytest.mark.parametrize("content_list, version_files", [
    (['model/3/', 'model/3.txt', 'model/3/2/'],
     (None, None, None)),
    (['model/3/model.xml', 'model/3/model.bin', 'model/3/sub/file'],
     ('s3://bucket/model/3/model.xml', 's3://bucket/model/3/model.bin', None)),
    (['model/3/something.xml', 'model/3/something.bin',
      'model/3/mapping_config.json'],
     ('s3://bucket/model/3/something.xml',
      's3://bucket/model/3/something.bin',
      's3://bucket/model/3/mapping_config.json')),
    (['model/3//file.bin', 'model/3//file.xml'],
     (None, None, None)),
    (['model/3/some file.bin', 'model/3/some file.xml'],
     ('s3://bucket/model/3/some file.xml', 's3://bucket/model/3/some file.bin',
      None)),
    (['model/3/somefile.bin', 'model/3/otherfile.xml'],
     (None, None, None)),
])
def test_get_versions_files(mocker, content_list, version_files):
    list_content_mocker = mocker.patch('ie_serving.models.s3_model.S3Model.'
                                       's3_list_content')
    list_content_mocker.return_value = content_list

    xml, bin, mapping = S3Model.get_version_files('s3://bucket/model/3/')
    assert (xml, bin, mapping) == version_files


def test_get_mapping_config(mocker):
    list_content_mocker = mocker.patch('ie_serving.models.s3_model.S3Model.'
                                       's3_list_content')
    list_content_mocker.return_value = ['model/3/doc.doc',
                                        'model/3/mapping_config.json',
                                        'model/3/model.xml',
                                        'model/3/model.bin']

    mapping = S3Model._get_mapping_config('s3://bucket/model/3/')
    assert mapping == 's3://bucket/model/3/mapping_config.json'


def test_not_mapping_config(mocker):
    list_content_mocker = mocker.patch('ie_serving.models.s3_model.S3Model.'
                                       's3_list_content')
    list_content_mocker.return_value = ['model/3/doc.doc',
                                        'model/3/config.json',
                                        'model/3/model.xml',
                                        'model/3/model.bin']

    mapping = S3Model._get_mapping_config('s3://bucket/model/3/')
    assert mapping is None
