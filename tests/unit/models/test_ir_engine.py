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

import json
from unittest import mock

import pytest
from conftest import MockedNet, MockedIOInfo

from ie_serving.models.ir_engine import IrEngine
from ie_serving.models.shape_management.batching_info import BatchingInfo
from ie_serving.models.shape_management.shape_info import ShapeInfo


def test_init_class():
    mapping_config = 'mapping_config.json'
    exec_net = None
    net = MockedNet(inputs={'input': MockedIOInfo('FP32', [1, 1, 1], 'NCHW')},
                    outputs={'output': MockedIOInfo('FP32', [1, 1], 'NCHW')})
    batching_info = BatchingInfo(None)
    shape_info = ShapeInfo(None, net.inputs)
    plugin = None
    engine = IrEngine(model_name='test', model_version=1,
                      mapping_config=mapping_config,
                      exec_net=exec_net,
                      net=net, plugin=plugin, batching_info=batching_info,
                      shape_info=shape_info)
    assert exec_net == engine.exec_net
    assert list(net.inputs.keys()) == engine.input_tensor_names
    assert list(net.outputs.keys()) == engine.output_tensor_names


def test_build_device_cpu(mocker):
    mocker.patch("ie_serving.models.ir_engine.IEPlugin")
    cpu_extension_mock = mocker.patch(
        "ie_serving.models.ir_engine.IEPlugin.add_cpu_extension")
    model_xml = 'model1.xml'
    model_bin = 'model1.bin'
    batch_size_param, shape_param = None, None
    mapping_config = 'mapping_config.json'
    with pytest.raises(Exception):
        IrEngine.build(model_name='test', model_version=1,
                       model_bin=model_bin, model_xml=model_xml,
                       mapping_config=mapping_config,
                       batch_size_param=batch_size_param,
                       shape_param=shape_param)
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
    batch_size_param, shape_param = None, None
    with pytest.raises(Exception):
        IrEngine.build(model_name='test', model_version=1,
                       model_bin=model_bin, model_xml=model_xml,
                       mapping_config=mapping_config,
                       batch_size_param=batch_size_param,
                       shape_param=shape_param)
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
