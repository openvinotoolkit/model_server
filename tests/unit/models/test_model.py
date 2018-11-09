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

import pytest
from ie_serving.models.model import Model


def test_model_init():
    new_model = Model(model_name="test", model_directory='fake_path',
                      available_versions=[1, 2, 3], engines={})
    assert new_model.default_version == 3
    assert new_model.model_name == 'test'
    assert new_model.model_directory == 'fake_path'
    assert new_model.engines == {}


def test_get_engines_for_model(mocker):
    engines_mocker = mocker.patch('ie_serving.models.ir_engine.IrEngine.'
                                  'build')
    engines_mocker.side_effect = ['modelv2', 'modelv4']
    available_versions = [{'xml_model_path': 'modelv2.xml',
                           'bin_model_path': 'modelv2.bin', 'version': 2},
                          {'xml_model_path': 'modelv4.xml', 'bin_model_path':
                           'modelv4.bin', 'version': 4}]
    output = Model.get_engines_for_model(versions=available_versions)
    assert 2 == len(output)
    assert 'modelv2' == output[2]
    assert 'modelv4' == output[4]


def test_get_engines_for_model_with_ir_raises(mocker):
    engines_mocker = mocker.patch('ie_serving.models.ir_engine.IrEngine.'
                                  'build')
    engines_mocker.side_effect = ['modelv2', 'modelv4', Exception("test")]
    available_versions = [{'xml_model_path': 'modelv2.xml',
                           'bin_model_path': 'modelv2.bin', 'version': 2},
                          {'xml_model_path': 'modelv4.xml', 'bin_model_path':
                              'modelv4.bin', 'version': 3},
                          {'xml_model_path': 'modelv4.xml', 'bin_model_path':
                              'modelv4.bin', 'version': 4}]
    output = Model.get_engines_for_model(versions=available_versions)
    assert 2 == len(output)
    assert 'modelv2' == output[2]
    assert 'modelv4' == output[3]
