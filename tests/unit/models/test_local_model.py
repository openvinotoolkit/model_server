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
from ie_serving.models.local_model import LocalModel
from ie_serving.models.model_version_status import ModelVersionStatus
from ie_serving.models.models_utils import ModelVersionState, ErrorCode


@pytest.mark.parametrize("engines", [
    {1: None, 2: None, 3: None},
    {1: None, 2: None},
    {}
])
def test_model_init(engines):
    available_versions = [1, 2, 3]
    model_name = "test"
    versions_statuses = {}
    for version in available_versions:
        versions_statuses[version] = ModelVersionStatus(model_name, version)

    for version_status in versions_statuses.values():
        assert version_status.state == ModelVersionState.START

    new_model = LocalModel(model_name=model_name, model_directory='fake_path',
                           available_versions=available_versions,
                           engines=engines,
                           batch_size_param=None,
                           shape_param=None,
                           version_policy_filter=lambda versions: versions[:],
                           versions_statuses=versions_statuses,
                           update_locks={},
                           num_ireq=1, target_device='CPU',
                           plugin_config=None)

    not_available_versions = list(set(available_versions) ^
                                  set(engines.keys()))
    for loaded_version in engines.keys():
        assert new_model.versions_statuses[loaded_version].state == \
            ModelVersionState.AVAILABLE
    for not_available_version in not_available_versions:
        assert new_model.versions_statuses[not_available_version].state != \
            ModelVersionState.AVAILABLE

    assert new_model.default_version == 3
    assert new_model.model_name == 'test'
    assert new_model.model_directory == 'fake_path'
    assert new_model.engines == engines


@pytest.mark.parametrize("mocker_values, expected_output", [
    ([['/data/model/3/model.bin'], ['/data/model/3/model.xml'], []],
     ['/data/model/3/model.xml', '/data/model/3/model.bin', None]),
    ([['/data/model/3/model_binary.bin'], ['/data/model/3/model_binary.xml'],
      []], ['/data/model/3/model_binary.xml', '/data/model/3/model_binary.bin',
            None]),
    ([['/data/model/3/model_xml.bin'], ['/data/model/3/model_xml.xml'], []],
     ['/data/model/3/model_xml.xml', '/data/model/3/model_xml.bin', None]),
    ([['/data/model/3/model.xml.bin'], ['/data/model/3/model.xml.xml'], []],
     ['/data/model/3/model.xml.xml', '/data/model/3/model.xml.bin', None]),
    ([[], ['/data/model/3/model.xml'], []],
     [None, None, None]),
    ([['/data/model/3/model.bin'], [], []],
     [None, None, None]),
    ([[], [], []],
     [None, None, None])
])
def test_get_versions_files(mocker, mocker_values, expected_output):
    glob_mocker = mocker.patch('glob.glob')
    glob_mocker.side_effect = mocker_values

    xml_f, bin_f, mapping = LocalModel.get_version_files('/data/model/3/')
    assert expected_output[0] == xml_f
    assert expected_output[1] == bin_f
    assert expected_output[2] is mapping


@pytest.mark.parametrize("is_error", [False, True])
def test_get_engines_for_model(mocker, is_error):
    engines_mocker = mocker.patch('ie_serving.models.ir_engine.IrEngine.'
                                  'build')

    engines_mocker.side_effect = ['modelv2', 'modelv4']
    available_versions = [{'xml_file': 'modelv2.xml',
                           'bin_file': 'modelv2.bin',
                           'mapping_config': 'mapping_config.json',
                           'version_number': 2, 'batch_size_param': None,
                           'shape_param': None, 'num_ireq': 1,
                           'target_device': 'CPU', 'plugin_config': None},
                          {'xml_file': 'modelv4.xml',
                           'bin_file': 'modelv4.bin',
                           'mapping_config': 'mapping_config.json',
                           'version_number': 4, 'batch_size_param': None,
                           'shape_param': None, 'num_ireq': 1,
                           'target_device': 'CPU', 'plugin_config': None}]
    versions_statuses = {}
    for version in available_versions:
        version_number = version['version_number']
        versions_statuses[version_number] = ModelVersionStatus("test",
                                                               version_number)
    if is_error:
        get_engine_for_version_mocker = mocker.patch(
            'ie_serving.models.local_model.LocalModel.get_engine_for_version')
        get_engine_for_version_mocker.side_effect = Exception()

    output = LocalModel.get_engines_for_model(
        model_name='test',
        versions_attributes=available_versions,
        versions_statuses=versions_statuses,
        update_locks={})

    for version_status in versions_statuses.values():
        assert version_status.state == ModelVersionState.LOADING
        if is_error:
            assert version_status.status['error_code'] == ErrorCode.UNKNOWN
            assert get_engine_for_version_mocker.called
        else:
            assert version_status.status['error_code'] == ErrorCode.OK
    if is_error:
        assert 0 == len(output)
    else:
        assert 2 == len(output)
        assert 'modelv2' == output[2]
        assert 'modelv4' == output[4]


def test_get_engines_for_model_with_ir_raises(mocker):
    engines_mocker = mocker.patch('ie_serving.models.ir_engine.IrEngine.'
                                  'build')
    engines_mocker.side_effect = ['modelv2', 'modelv4', Exception("test")]
    available_versions = [{'xml_file': 'modelv2.xml',
                           'bin_file': 'modelv2.bin',
                           'mapping_config': 'mapping_config.json',
                           'version_number': 2, 'batch_size_param': None,
                           'shape_param': None, 'num_ireq': 1,
                           'target_device': 'CPU', 'plugin_config': None},
                          {'xml_file': 'modelv4.xml',
                           'bin_file': 'modelv4.bin',
                           'mapping_config': 'mapping_config.json',
                           'version_number': 3, 'batch_size_param': None,
                           'shape_param': None, 'num_ireq': 1,
                           'target_device': 'CPU', 'plugin_config': None},
                          {'xml_file': 'modelv4.xml',
                           'bin_file': 'modelv4.bin',
                           'mapping_config': 'mapping_config.json',
                           'version_number': 4, 'batch_size_param': None,
                           'shape_param': None, 'num_ireq': 1,
                           'target_device': 'CPU', 'plugin_config': None}]
    versions_statuses = {}
    for version in available_versions:
        version_number = version['version_number']
        versions_statuses[version_number] = ModelVersionStatus(
            "test", version_number)
    output = LocalModel.get_engines_for_model(
        model_name='test',
        versions_attributes=available_versions,
        versions_statuses=versions_statuses,
        update_locks={})
    assert 2 == len(output)
    assert 'modelv2' == output[2]
    assert 'modelv4' == output[3]


def test_get_versions():
    model = LocalModel
    file_path = os.path.realpath(__file__)
    unit_tests_path = os.path.dirname(os.path.dirname(file_path))
    output = model.get_versions(unit_tests_path)
    assert 3 == len(output)
    output = model.get_versions(unit_tests_path + os.sep)
    assert 3 == len(output)
