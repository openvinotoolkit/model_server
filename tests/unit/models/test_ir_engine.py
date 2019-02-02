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

from ie_serving.models.ir_engine import IrEngine
from unittest import mock
import json
import pytest
from conftest import Layer


def test_init_class():
    model_xml = 'model1.xml'
    model_bin = 'model1.bin'
    mapping_config = 'mapping_config.json'
    exec_net = None
    net = None
    batch_size = None
    plugin = None
    input_key = 'input'
    inputs = {input_key: Layer('FP32', (1, 1), 'NCHW')}
    outputs = {'output': Layer('FP32', (1, 1), 'NCHW')}
    engine = IrEngine(model_bin=model_bin, model_xml=model_xml,
                      mapping_config=mapping_config, exec_net=exec_net,
                      inputs=inputs, outputs=outputs, net=net, plugin=plugin,
                      batch_size=batch_size)
    assert model_xml == engine.model_xml
    assert model_bin == engine.model_bin
    assert exec_net == engine.exec_net
    assert [input_key] == engine.input_tensor_names
    assert inputs == engine.input_tensors
    assert ['output'] == engine.output_tensor_names
    assert {'inputs': {'input': 'input'},
            'outputs': {'output': 'output'}} == engine.model_keys
    assert [input_key] == engine.input_key_names


def test_build_device_cpu(mocker):
    mocker.patch("ie_serving.models.ir_engine.IEPlugin")
    cpu_extension_mock = mocker.patch(
        "ie_serving.models.ir_engine.IEPlugin.add_cpu_extension")
    model_xml = 'model1.xml'
    model_bin = 'model1.bin'
    batch_size = None
    mapping_config = 'mapping_config.json'
    with pytest.raises(Exception):
        IrEngine.build(model_bin=model_bin, model_xml=model_xml,
                       mapping_config=mapping_config,
                       batch_size=batch_size)
        cpu_extension_mock.assert_called_once_with()


def test_build_device_other(mocker):
    mocker.patch("ie_serving.models.ir_engine.IEPlugin")
    device_mocker = mocker.patch("ie_serving.models.ir_engine.DEVICE")
    device_mocker.return_value = 'other'
    cpu_extension_mock = mocker.patch(
        "ie_serving.models.ir_engine.IEPlugin.add_cpu_extension")
    model_xml = 'model1.xml'
    model_bin = 'model1.bin'
    mapping_config = 'mapping_config.json'
    batch_size = None
    with pytest.raises(Exception):
        IrEngine.build(model_bin=model_bin, model_xml=model_xml,
                       mapping_config=mapping_config,
                       batch_size=batch_size)
        assert not cpu_extension_mock.assert_called_once_with()


def test_mapping_config_not_exists(get_fake_ir_engine):
    engine = get_fake_ir_engine
    output = engine._get_mapping_data_if_exists('mapping_config.json')
    assert None is output


def test_mapping_config_exists_ok(mocker, get_fake_ir_engine):
    test_dict = {'config': 'test'}
    test_json = json.dumps(test_dict)
    mocker.patch("ie_serving.models.ir_engine.open",
                 new=mock.mock_open(read_data=test_json))
    glob_glob_mocker = mocker.patch('glob.glob')
    glob_glob_mocker.return_value = ['fake_path']
    engine = get_fake_ir_engine
    output = engine._get_mapping_data_if_exists('mapping_config.json')
    assert test_dict == output


def test_mapping_config_exists_cannot_open_file(mocker, get_fake_ir_engine):
    glob_glob_mocker = mocker.patch('glob.glob')
    glob_glob_mocker.return_value = ['fake_path']
    engine = get_fake_ir_engine
    output = engine._get_mapping_data_if_exists('mapping_config.json')
    assert None is output


def test_mapping_config_exists_cannot_load_json(mocker, get_fake_ir_engine):
    test_data = "not json"
    mocker.patch("ie_serving.models.ir_engine.open",
                 new=mock.mock_open(read_data=test_data))
    glob_glob_mocker = mocker.patch('glob.glob')
    glob_glob_mocker.return_value = ['fake_path']
    glob_glob_mocker = mocker.patch('glob.glob')
    glob_glob_mocker.return_value = ['fake_path']
    engine = get_fake_ir_engine
    output = engine._get_mapping_data_if_exists('mapping_config.json')
    assert None is output


def test_set_tensor_names_as_keys(get_fake_ir_engine):
    engine = get_fake_ir_engine
    expected_output = {'inputs': {'input': 'input'},
                       'outputs': {'output': 'output'}}
    output = engine._set_tensor_names_as_keys()
    assert output == expected_output


@pytest.mark.parametrize("input_data, tensors, expected_output", [
    ({"wrong": {"input": "test_input"}}, ['input'], {"input": "input"}),
    ({"inputs": {"input": "test_input"}}, ['input'], {"test_input": "input"}),
    ({"test": {"input": "test_input"}}, ['input'], {"input": "input"}),
    ({"inputs": {"input": "test_input"}}, ['input', 'input2'],
     {"test_input": "input", "input2": "input2"}),
    ({"inputs": {"input": "test_input", "in": 'test'}}, ['input', 'input2'],
     {"test_input": "input", "input2": "input2"}),
    ({"inputs": {"input": "test_input", 'input2': "in"}}, ['input', 'input2'],
     {"test_input": "input", "in": "input2"})
])
def test_return_proper_key_value(get_fake_ir_engine, input_data, tensors,
                                 expected_output):
    which_way = 'inputs'
    engine = get_fake_ir_engine
    output = engine._return_proper_key_value(data=input_data, tensors=tensors,
                                             which_way=which_way)
    assert expected_output == output


def test_set_names_in_config_as_keys(get_fake_ir_engine, mocker):
    engine = get_fake_ir_engine
    key_value_mocker = mocker.patch('ie_serving.models.'
                                    'ir_engine.IrEngine.'
                                    '_return_proper_key_value')
    key_value_mocker.side_effect = ['test', 'test']
    output = engine._set_names_in_config_as_keys(data={})

    assert {'inputs': 'test', 'outputs': 'test'} == output


def test_set_keys(get_fake_ir_engine, mocker):
    engine = get_fake_ir_engine
    get_config_file_mocker = mocker.patch('ie_serving.models.'
                                          'ir_engine.IrEngine.'
                                          '_get_mapping_data_if_exists')
    get_config_file_mocker.side_effect = [None, 'something']

    tensor_names_as_keys_mocker = mocker.patch('ie_serving.models.'
                                               'ir_engine.IrEngine.'
                                               '_set_tensor_names_as_keys')
    tensor_names_as_keys_mocker.return_value = 'tensor_name'

    keys_from_config_mocker = mocker.patch('ie_serving.models.'
                                           'ir_engine.IrEngine.'
                                           '_set_names_in_config_as_keys')
    keys_from_config_mocker.return_value = 'config'

    output = engine.set_keys('mapping_config.json')
    tensor_names_as_keys_mocker.assert_called_once_with()
    assert 'tensor_name' == output

    output = engine.set_keys('mapping_config.json')
    keys_from_config_mocker.assert_called_once_with('something')
    assert 'config' == output
