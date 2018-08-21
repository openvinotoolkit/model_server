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

from ie_serving.server import service_utils
import pytest


@pytest.mark.parametrize("requested_model, requested_ver, expected_ver, "
                         "expected_validation",
                         [('resnet', 1, 1, True), ('resnet', 0, 8, True),
                          ('resnet', 444, 0, False), ('Xception', 1, 1, True),
                          ('xception', 0, 0, False), ('xception', 6, 0, False)
                          ])
def test_check_availability_of_requested_model(mocker, requested_model,
                                               requested_ver, expected_ver,
                                               expected_validation):

    resnet_model_object = {'name': 'resnet', 'versions': [1, 2, 8],
                           'default_version': 8}
    inception_model_object = {'name': 'inception', 'versions': [3, 4, 5],
                              'default_version': 5}
    xception_model_object = {'name': 'Xception', 'versions': [1, 6, 8],
                             'default_version': 8}
    models = [resnet_model_object, inception_model_object,
              xception_model_object]

    available_models = {"resnet": None, 'inception': None, 'Xception': None}
    for x in models:
        model_mocker = mocker.patch('ie_serving.models.model.Model')
        model_mocker.versions = x['versions']
        model_mocker.default_version = x['default_version']
        available_models[x['name']] = model_mocker

    validation, version = service_utils.check_availability_of_requested_model(
        models=available_models, model_name=requested_model,
        requested_version=requested_ver)
    assert expected_validation == validation
    assert expected_ver == version
