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
from ie_serving.models.s3_model import S3Model


def test_get_versions(mocker):
    list_content_mocker = mocker.patch('ie_serving.models.s3_model.S3Model.'
                                       's3_list_content')
    list_content_mocker.return_value = ['model/3/',
                                        'model/3.txt',
                                        'model/sub/2/',
                                        'model/one']

    output = S3Model.get_versions('s3://bucket/model')
    assert 1 == len(output)
    assert 's3://bucket/model/3/' == output[0]


def test_not_get_versions_files(mocker):
    list_content_mocker = mocker.patch('ie_serving.models.s3_model.S3Model.'
                                       's3_list_content')
    list_content_mocker.return_value = ['model/3/doc.doc',
                                        'model/3/subdir',
                                        'model/3/something.xml',
                                        'model/3/model.bin']

    xml, bin, mapping = S3Model.get_version_files('s3://bucket/model/3/')
    assert xml is None and bin is None and mapping is None


def test_get_versions_files(mocker):
    list_content_mocker = mocker.patch('ie_serving.models.s3_model.S3Model.'
                                       's3_list_content')
    list_content_mocker.return_value = ['model/3/doc.doc',
                                        'model/3/subdir',
                                        'model/3/model.xml',
                                        'model/3/model.bin']

    xml, bin, mapping = S3Model.get_version_files('s3://bucket/model/3/')
    assert xml == 's3://bucket/model/3/model.xml' and \
        bin == 's3://bucket/model/3/model.bin' and \
        mapping is None
