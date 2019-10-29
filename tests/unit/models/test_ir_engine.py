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
import datetime
import json
import queue
from unittest import mock

import pytest
from config import RESHAPE_TEST_CASES, \
    SCAN_INPUT_SHAPES_TEST_CASES, DETECT_SHAPES_INCOMPATIBILITY_TEST_CASES
from conftest import MockedNet, MockedIOInfo

from ie_serving.models import InferenceStatus
from ie_serving.models.ir_engine import IrEngine, inference_callback
from ie_serving.models.shape_management.batching_info import BatchingInfo
from ie_serving.models.shape_management.shape_info import ShapeInfo
from ie_serving.server.request import Request


@pytest.mark.parametrize("status", [InferenceStatus.OK, InferenceStatus.ERROR])
def test_inference_callback(get_fake_ir_engine, status):
    py_data = {
        'ir_engine': get_fake_ir_engine,
        'request': Request({}),
        'ireq_index': 0,
        'start_time': datetime.datetime.now()
    }
    inference_callback(status, py_data)
    if status == InferenceStatus.OK:
        assert py_data['request'].result == {}
    else:
        assert py_data['request'].result == \
               "Error occurred during inference execution"


def test_init_class():
    mapping_config = 'mapping_config.json'
    exec_net = None
    net = MockedNet(inputs={'input': MockedIOInfo('FP32', [1, 1, 1], 'NCHW')},
                    outputs={'output': MockedIOInfo('FP32', [1, 1], 'NCHW')})
    batching_info = BatchingInfo(None)
    shape_info = ShapeInfo(None, net.inputs)
    plugin = None
    requests_queue = queue.Queue()
    free_ireq_index_queue = queue.Queue(maxsize=1)
    free_ireq_index_queue.put(0)
    engine = IrEngine(model_name='test', model_version=1,
                      mapping_config=mapping_config,
                      exec_net=exec_net,
                      net=net, plugin=plugin, batching_info=batching_info,
                      shape_info=shape_info, num_ireq=1,
                      free_ireq_index_queue=free_ireq_index_queue,
                      requests_queue=requests_queue,
                      target_device='CPU',
                      plugin_config=None)
    assert exec_net == engine.exec_net
    assert list(net.inputs.keys()) == engine.input_tensor_names
    assert list(net.outputs.keys()) == engine.output_tensor_names
    assert engine.free_ireq_index_queue.qsize() == 1


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
                       shape_param=shape_param, num_ireq=1,
                       target_device='CPU', plugin_config=None)
        cpu_extension_mock.assert_called_once_with()


def test_build_device_other(mocker):
    mocker.patch("ie_serving.models.ir_engine.IEPlugin")
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
                       shape_param=shape_param, num_ireq=1,
                       target_device='other', plugin_config=None)
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


@pytest.mark.parametrize("shape_mode, changed_inputs, expected_reshape_param",
                         DETECT_SHAPES_INCOMPATIBILITY_TEST_CASES)
def test_detect_shapes_incompatibility(get_fake_ir_engine, mocker,
                                       shape_mode, changed_inputs,
                                       expected_reshape_param):
    engine = get_fake_ir_engine
    engine.shape_info.mode = shape_mode
    scan_input_shapes_mock = mocker.patch('ie_serving.models.'
                                          'ir_engine.IrEngine.'
                                          'scan_input_shapes')
    scan_input_shapes_mock.return_value = changed_inputs
    reshape_param = engine.detect_shapes_incompatibility(None)
    assert reshape_param == expected_reshape_param


@pytest.mark.parametrize("net_inputs_shapes, data, expected_output",
                         SCAN_INPUT_SHAPES_TEST_CASES)
def test_scan_input_shapes(get_fake_ir_engine, net_inputs_shapes, data,
                           expected_output):
    engine = get_fake_ir_engine

    # update network with desired inputs
    new_inputs = {}
    for input_name, input_shape in net_inputs_shapes.items():
        new_inputs.update({input_name: MockedIOInfo('FP32', list(input_shape),
                                                    'NCHW')})
    engine.net.inputs = new_inputs

    output = engine.scan_input_shapes(data)
    assert output == expected_output


@pytest.mark.parametrize("reshape_param, calls_config, returns_config, "
                         "expected_output", RESHAPE_TEST_CASES)
def test_reshape(get_fake_ir_engine, mocker, reshape_param, calls_config,
                 returns_config, expected_output):
    engine = get_fake_ir_engine
    methods_mocks = {
        method_name: mocker.patch('ie_serving.models.ir_engine.IrEngine.{}'
                                  .format(method_name))
        for method_name in list(calls_config.keys())
    }

    for method_name, return_value in returns_config.items():
        methods_mocks[method_name].return_value = return_value

    output = engine.reshape(reshape_param)

    for method_name, is_called in calls_config.items():
        assert methods_mocks[method_name].called == is_called
    assert output == expected_output
