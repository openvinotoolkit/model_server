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

import grpc
import numpy as np
from tensorflow import make_ndarray
from conftest import get_fake_request, PREDICT_SERVICE, make_tensor_proto, \
    predict_pb2


def test_predict_successful(mocker, get_grpc_service_for_predict,
                            get_fake_model):
    results_mock = mocker.patch(
        'ie_serving.server.request.Request.wait_for_result')
    expected_response = np.ones(shape=(2, 2))
    results_mock.return_value = ({'output': expected_response}, 0)

    request = get_fake_request(model_name='test',
                               data_shape=(1, 1, 1), input_blob='input')
    grpc_server = get_grpc_service_for_predict
    rpc = grpc_server.invoke_unary_unary(
            PREDICT_SERVICE.methods_by_name['Predict'],
            (),
            request, None)
    rpc.initial_metadata()
    response, trailing_metadata, code, details = rpc.termination()

    encoded_response = make_ndarray(response.outputs['output'])
    assert get_fake_model.default_version == response.model_spec.version.value
    assert grpc.StatusCode.OK == code
    assert expected_response.shape == encoded_response.shape


def test_predict_successful_version(mocker, get_grpc_service_for_predict):
    results_mock = mocker.patch(
        'ie_serving.server.request.Request.wait_for_result')
    expected_response = np.ones(shape=(2, 2))
    results_mock.return_value = ({'output': expected_response}, None)
    requested_version = 1
    request = get_fake_request(model_name='test', data_shape=(1, 1, 1),
                               input_blob='input', version=requested_version)
    grpc_server = get_grpc_service_for_predict
    rpc = grpc_server.invoke_unary_unary(
        PREDICT_SERVICE.methods_by_name['Predict'],
        (),
        request, None)
    rpc.initial_metadata()
    response, trailing_metadata, code, details = rpc.termination()

    encoded_response = make_ndarray(response.outputs['output'])
    assert requested_version == response.model_spec.version.value
    assert grpc.StatusCode.OK == code
    assert expected_response.shape == encoded_response.shape


def test_predict_wrong_model_name(get_grpc_service_for_predict):
    wrong_model_name = 'wrong_name'
    request = get_fake_request(model_name=wrong_model_name, data_shape=(1, 1),
                               input_blob='input')
    grpc_server = get_grpc_service_for_predict
    rpc = grpc_server.invoke_unary_unary(
        PREDICT_SERVICE.methods_by_name['Predict'],
        (),
        request, None)
    rpc.initial_metadata()
    response, trailing_metadata, code, details = rpc.termination()
    assert grpc.StatusCode.NOT_FOUND == code


def test_predict_wrong_model_version(get_grpc_service_for_predict):
    wrong_requested_version = 999
    request = get_fake_request(model_name='test', data_shape=(1, 1),
                               input_blob='input',
                               version=wrong_requested_version)
    grpc_server = get_grpc_service_for_predict
    rpc = grpc_server.invoke_unary_unary(
        PREDICT_SERVICE.methods_by_name['Predict'],
        (),
        request, None)
    rpc.initial_metadata()
    response, trailing_metadata, code, details = rpc.termination()
    assert grpc.StatusCode.NOT_FOUND == code


def test_predict_wrong_shape(get_grpc_service_for_predict):
    wrong_shape = (4, 4)
    request = get_fake_request(model_name='test', data_shape=wrong_shape,
                               input_blob='input')
    grpc_server = get_grpc_service_for_predict
    rpc = grpc_server.invoke_unary_unary(
        PREDICT_SERVICE.methods_by_name['Predict'],
        (),
        request, None)
    rpc.initial_metadata()
    response, trailing_metadata, code, details = rpc.termination()
    assert grpc.StatusCode.INVALID_ARGUMENT == code


def test_predict_wrong_input_blob(get_grpc_service_for_predict):
    wrong_input_blob = 'wrong_input_blob'
    request = get_fake_request(model_name='test', data_shape=(1, 1),
                               input_blob=wrong_input_blob)
    grpc_server = get_grpc_service_for_predict
    rpc = grpc_server.invoke_unary_unary(
          PREDICT_SERVICE.methods_by_name['Predict'],
          (),
          request, None)
    rpc.initial_metadata()
    response, trailing_metadata, code, details = rpc.termination()
    assert grpc.StatusCode.INVALID_ARGUMENT == code


def test_predict_problem_with_serialize_data(get_grpc_service_for_predict):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'test'
    data = np.ones(shape=(2, 2, 2))
    request.inputs['input'].CopyFrom(
        make_tensor_proto(data, shape=(3, 3, 3)))

    grpc_server = get_grpc_service_for_predict
    rpc = grpc_server.invoke_unary_unary(
        PREDICT_SERVICE.methods_by_name['Predict'],
        (),
        request, None)
    rpc.initial_metadata()
    response, trailing_metadata, code, details = rpc.termination()
    assert grpc.StatusCode.INVALID_ARGUMENT == code
