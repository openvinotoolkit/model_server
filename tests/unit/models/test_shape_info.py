#
# Copyright (c) 2019 Intel Corporation
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

from ie_serving.models.shape_management.shape_info import ShapeInfo
from config import PROCESS_SHAPE_PARAM_TEST_CASES, \
    PROCESS_GET_SHAPE_FROM_STRING_TEST_CASES, NOT_CALLED


@pytest.mark.parametrize("shape_param, net_inputs, calls_config, "
                         "returns_config, expected_output",
                         PROCESS_SHAPE_PARAM_TEST_CASES)
def test_process_shape_param(mocker, shape_param, net_inputs, calls_config,
                             returns_config, expected_output):
    methods_mocks = {
        method_name: mocker.patch('ie_serving.models.shape_management.'
                                  'shape_info.ShapeInfo.{}'
                                  .format(method_name))
        for method_name in list(calls_config.keys())
    }

    for method_name, return_value in returns_config.items():
        methods_mocks[method_name].return_value = return_value

    shape_info = ShapeInfo(None, None)
    output = shape_info.process_shape_param(shape_param, net_inputs)
    for method_name, is_called in calls_config.items():
        assert methods_mocks[method_name].called == is_called
    assert output == expected_output


@pytest.mark.parametrize("shape, net_inputs, is_error", [
    ((1, 1, 1), {"in": (1, 1, 1)}, False),
    ((1, 1, 1), {"in": (1, 1, 1), "in2": (1, 1, 1)}, True)
])
def test_shape_as_dict(shape: tuple, net_inputs: dict, is_error):
    shape_info = ShapeInfo(None, None)
    if is_error:
        with pytest.raises(Exception):
            shape_info._shape_as_dict(shape, net_inputs)
    else:
        output = shape_info._shape_as_dict(shape, net_inputs)
        assert output == {"in": shape}


@pytest.mark.parametrize("shape_param, calls_config, "
                         "returns_config, expected_output",
                         PROCESS_GET_SHAPE_FROM_STRING_TEST_CASES)
def test_get_shape_from_string(mocker, shape_param, calls_config,
                               returns_config, expected_output):
    methods_mocks = {
        method_name: mocker.patch('ie_serving.models.shape_management.'
                                  'shape_info.ShapeInfo.{}'
                                  .format(method_name))
        for method_name in list(calls_config.keys())
    }

    for method_name, return_value in returns_config.items():
        methods_mocks[method_name].return_value = return_value

    shape_info = ShapeInfo(None, None)
    output = shape_info.get_shape_from_string(shape_param)
    for method_name, is_called in calls_config.items():
        assert methods_mocks[method_name].called == is_called
    assert output == expected_output


@pytest.mark.parametrize("shapes, expected_output", [
    ({"input": "(1,1,1)"}, {"input": (1, 1, 1)}),
    ({1: "(1,1,1)"}, None),
    ({"input": 1}, None),
    ({"input": "(1,1,1)", 1: "(1,1,1)", "in": 1}, {"input": (1, 1, 1)})
])
def test_get_shape_dict(shapes, expected_output):
    shape_info = ShapeInfo(None, None)
    output = shape_info.get_shape_dict(shapes)
    assert output == expected_output


@pytest.mark.parametrize("input_name, shape, load_shape_result, "
                         "get_shape_tuple_called, get_shape_tuple_result, "
                         "expected_output",
                         [
                             ("input", "string", None, False, NOT_CALLED, {}),
                             ("input", "[\"string\",1,1]", ["string", 1, 1],
                              True, None, {}),
                             ("input", "[1,1,1]", [1, 1, 1], True,
                              (1, 1, 1), {"input": (1, 1, 1)})
                         ])
def test_get_single_shape(mocker, input_name, shape, load_shape_result,
                          get_shape_tuple_called, get_shape_tuple_result,
                          expected_output):
    shape_info = ShapeInfo(None, None)
    load_shape_mock = mocker.patch('ie_serving.models.shape_management.'
                                   'shape_info.ShapeInfo.load_shape')
    get_shape_tuple_mock = mocker.patch('ie_serving.models.shape_management.'
                                        'shape_info.ShapeInfo.get_shape_tuple')
    load_shape_mock.return_value = load_shape_result
    get_shape_tuple_mock.return_value = get_shape_tuple_result

    output = shape_info._get_single_shape(input_name, shape)

    assert load_shape_mock.called
    assert get_shape_tuple_mock.called == get_shape_tuple_called
    assert output == expected_output


@pytest.mark.parametrize("shape, expected_output", [
    ([1, 1, 1], (1, 1, 1)),
    (["string", 1, 1], None),
])
def test_get_shape_tuple(shape, expected_output):
    shape_info = ShapeInfo(None, None)
    output = shape_info.get_shape_tuple(shape)
    assert output == expected_output


@pytest.mark.parametrize("shape, expected_output", [
    ("[1, 1, 1]", [1, 1, 1]),
    ("[\"string\", 1, 1]", ["string", 1, 1]),
    ("{\"input\": \"[1, 1, 1]\"}", {"input": "[1, 1, 1]"}),
    ("string", None),
    (1, None)
])
def test_load_shape(shape, expected_output):
    shape_info = ShapeInfo(None, None)
    output = shape_info.load_shape(shape)
    assert output == expected_output
