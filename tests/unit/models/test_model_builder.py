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
from ie_serving.models.model_builder import ModelBuilder


def test_build_local_model(mocker):
    local_build_mocker = mocker.patch(
        'ie_serving.models.local_model.LocalModel.build')
    ModelBuilder.build('model_name', 'opt/bucket/model', None, None, None, 1,
                       'CPU', None)
    assert local_build_mocker.called


def test_build_gs_model(mocker):
    gs_build_mocker = mocker.patch(
        'ie_serving.models.gs_model.GSModel.build')
    ModelBuilder.build('model_name', 'gs://bucket/model', None, None, None, 1,
                       'CPU', None)
    assert gs_build_mocker.called


def test_build_s3_model(mocker):
    s3_build_mocker = mocker.patch(
        'ie_serving.models.s3_model.S3Model.build')
    ModelBuilder.build('model_name', 's3://bucket/model', None, None, None, 1,
                       'CPU', None)
    assert s3_build_mocker.called
