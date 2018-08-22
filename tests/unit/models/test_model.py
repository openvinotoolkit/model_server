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
from ie_serving.models.model import Model


def test_model_init():
    new_model = Model(model_name="test", model_directory='fake_path',
                      available_versions=[1, 2, 3], engines={})
    assert new_model.default_version == 3
    assert new_model.model_name == 'test'
    assert new_model.model_directory == 'fake_path'
    assert new_model.engines == {}


@pytest.mark.parametrize("path, expected_value", [
    ('fake_path/model/1', 1),
    ('fake_path/model/1/test', 0),
    ('fake_path/model/56', 56)
])
def test_get_version_number_of_model(path, expected_value):
    output = Model.get_model_version_number(version_path=path)
    assert output == expected_value


@pytest.mark.parametrize("path, model_files, expected_value", [
    ('fake_path/model/1', [['model.bin'], ['model.xml']],
     ('model.xml', 'model.bin')),
    ('fake_path/model/1', [['model'], ['model.xml']],
     (None, None)),
    ('fake_path/model/1', [['model.bin'], ['model.yml']],
     (None, None))
])
def test_get_absolute_path_to_model(mocker, path, model_files,
                                    expected_value):
    model_mocker = mocker.patch('glob.glob')
    model_mocker.side_effect = model_files
    output1, output2 = Model.get_absolute_path_to_model(
        specific_version_model_path=path)
    assert expected_value[0] == output1
    assert expected_value[1] == output2


def test_get_all_available_versions(mocker):
    new_model = Model(model_name="test", model_directory='fake_path/model/',
                      available_versions=[1, 2, 3], engines={})
    model_mocker = mocker.patch('glob.glob')
    models_path = [new_model.model_directory + str(x) for x in range(5)]
    model_mocker.return_value = models_path
    absolute_path_model_mocker = mocker.patch('ie_serving.models.model.Model.'
                                              'get_absolute_path_to_model')
    absolute_path_model_mocker.side_effect = [(None, None),
                                              ('modelv2.xml', 'modelv2.bin'),
                                              (None, None),
                                              ('modelv4.xml', 'modelv4.bin')]
    output = new_model.get_all_available_versions(new_model.model_directory)
    expected_output = [{'xml_model_path': 'modelv2.xml',
                        'bin_model_path': 'modelv2.bin', 'version': 2},
                       {'xml_model_path': 'modelv4.xml', 'bin_model_path':
                           'modelv4.bin', 'version': 4}]

    assert 2 == len(output)
    assert expected_output == output


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
