#
# Copyright (c) 2021 Intel Corporation
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

from grpc._channel import _RPCState, _InactiveRpcError, Channel
import pytest
import numpy as np
from ovmsclient.tfs_compat.grpc.requests import GrpcPredictRequest

from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.get_model_metadata_pb2 import GetModelMetadataRequest # noqa
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.model_service_pb2_grpc import ModelServiceStub # noqa
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.predict_pb2 import (PredictRequest,
                                                                              PredictResponse)
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub # noqa
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusRequest, ModelVersionStatus # noqa
from ovmsclient.tfs_compat.protos.tensorflow.core.protobuf.error_codes_pb2 import Code as ErrorCode
from ovmsclient.tfs_compat.protos.tensorflow.core.framework.types_pb2 import DataType

from ovmsclient.tfs_compat.grpc.serving_client import make_grpc_client
from ovmsclient.tfs_compat.grpc.requests import GrpcModelStatusRequest, GrpcModelMetadataRequest
from ovmsclient.tfs_compat.grpc.responses import GrpcModelMetadataResponse, GrpcPredictResponse
from ovmsclient.tfs_compat.base.errors import BadResponseError

from config import (MODEL_STATUS_INVALID_PARAMS, PREDICT_INVALID_PARAMS,
                    MODEL_METADATA_INVALID_PARAMS)
from tfs_compat_grpc.config import (BUILD_INVALID_CONFIG, BUILD_VALID, BUILD_INVALID_CERTS,
                                    MODEL_METADATA_RESPONSE_VALID,
                                    COMMON_INVALID_GRPC,
                                    PREDICT_RESPONSE_VALID,
                                    PREDICT_RESPONSE_TENSOR_TYPE_INVALID,
                                    PREDICT_INVALID_GRPC)

from tfs_compat_grpc.utils import (create_model_metadata_response,
                                   create_model_status_response,
                                   merge_model_status_responses)


@pytest.fixture
def valid_grpc_serving_client_min():
    return make_grpc_client("localhost:9000")


def create_model_status_request(name, version, raw_name, raw_version):
    raw_request = GetModelStatusRequest()
    raw_request.model_spec.name = raw_name
    raw_request.model_spec.version.value = raw_version
    return GrpcModelStatusRequest(name, version, raw_request)


@pytest.fixture
def valid_model_metadata_response():
    model_raw_metadata_response_dict = {
        'version': 2,
        'name': 'resnet',
        'inputs': {
            '0': {
                'shape': [1, 3, 244, 244],
                'dtype': DataType.DT_FLOAT
            }
        },
        'outputs': {
            '1463': {
                'shape': [1, 1000],
                'dtype': DataType.DT_FLOAT
            }
        }
    }
    raw_response = create_model_metadata_response(model_raw_metadata_response_dict)
    return GrpcModelMetadataResponse(raw_response)


def create_model_metadata_request(name, version, raw_name, raw_version,
                                  metadata_field_list):
    raw_request = GetModelMetadataRequest()
    raw_request.model_spec.name = raw_name
    raw_request.model_spec.version.value = raw_version
    for field in metadata_field_list:
        raw_request.metadata_field.append(field)
    return GrpcModelMetadataRequest(name, version, raw_request)


def create_predict_response(outputs):
    predict_raw_response = PredictResponse()

    predict_raw_response.model_spec.name = 'name'
    predict_raw_response.model_spec.version.value = 0

    for key, value in outputs.items():
        predict_raw_response.outputs[key].CopyFrom(value)
    return GrpcPredictResponse(predict_raw_response)


def create_predict_request(name, version, raw_name, raw_version,
                           inputs_dict, raw_request_inputs_dict):
    raw_request = PredictRequest()
    raw_request.model_spec.name = raw_name
    raw_request.model_spec.version.value = raw_version
    for key, value in raw_request_inputs_dict.items():
        raw_request.inputs[key].CopyFrom(value)

    return GrpcPredictRequest(inputs_dict, name, version, raw_request)


def create_grpc_error(code, details):
    grpc_state = _RPCState(due=(), initial_metadata=None,
                           trailing_metadata=None, code=code,
                           details=details)
    grpc_error = _InactiveRpcError(grpc_state)
    return grpc_error


@pytest.mark.parametrize("config, method_call_count", BUILD_VALID)
def test_make_grpc_client_valid(mocker, config, method_call_count):
    mock_check_url = mocker.patch('ovmsclient.tfs_compat.base.serving_client'
                                  '.ServingClient._check_url')
    mock_check_tls_config = mocker.patch('ovmsclient.tfs_compat.base.serving_client'
                                         '.ServingClient._check_tls_config')
    mock_prepare_certs = mocker.patch('ovmsclient.tfs_compat.base.serving_client'
                                      '.ServingClient._prepare_certs',
                                      return_value=(b'server_certificate',
                                                    b'client_certificate',
                                                    b'client_key'))

    client = make_grpc_client(**config)

    assert mock_check_url.call_count == method_call_count['_check_url']
    assert mock_check_tls_config.call_count == method_call_count['_check_tls_config']
    assert mock_prepare_certs.call_count == method_call_count['_prepare_certs']
    assert type(client.channel) == Channel
    assert type(client.model_service_stub) == ModelServiceStub
    assert type(client.prediction_service_stub) == PredictionServiceStub


@pytest.mark.parametrize("config, method_call_spec, expected_exception, expected_message",
                         BUILD_INVALID_CONFIG)
def test_make_grpc_client_invalid_config(mocker, config, method_call_spec, expected_exception,
                                         expected_message):
    mocks = []
    for method_name, call_spec in method_call_spec.items():
        call_count, error_raised = call_spec
        mock = mocker.patch(f"ovmsclient.tfs_compat.base.serving_client."
                            f"ServingClient.{method_name}", side_effect=error_raised)
        mocks.append((mock, call_count))

    with pytest.raises(expected_exception) as e_info:
        make_grpc_client(**config)

    assert str(e_info.value) == expected_message
    for mock_info in mocks:
        mock, call_count = mock_info
        assert mock.call_count == call_count


@pytest.mark.parametrize("config, method_call_spec, expected_exception, expected_message",
                         BUILD_INVALID_CERTS)
def test_make_grpc_client_invalid_certs(mocker, config, method_call_spec, expected_exception,
                                        expected_message):
    mocks = []
    for method_name, call_spec in method_call_spec.items():
        call_count, error_raised = call_spec
        mock = mocker.patch(f"ovmsclient.tfs_compat.base.serving_client."
                            f"ServingClient.{method_name}", side_effect=error_raised)
        mocks.append((mock, call_count))

    with pytest.raises(expected_exception) as e_info:
        make_grpc_client(**config)

    assert str(e_info.value) == expected_message
    for mock_info in mocks:
        mock, call_count = mock_info
        assert mock.call_count == call_count


@pytest.mark.parametrize("params, expected_error, error_message", MODEL_STATUS_INVALID_PARAMS)
def test_get_model_status_invalid_params(mocker, valid_grpc_serving_client_min,
                                         params, expected_error, error_message):

    valid_grpc_serving_client_min.model_service_stub.GetModelStatus\
        = mocker.Mock()

    with pytest.raises(expected_error) as error:
        valid_grpc_serving_client_min.get_model_status(*params)

    valid_grpc_serving_client_min.model_service_stub.GetModelStatus.call_count == 0
    assert str(error.value) == error_message


def test_get_model_status_valid(mocker, valid_grpc_serving_client_min):
    raw_response = merge_model_status_responses([
        create_model_status_response(1, ErrorCode.OK, "OK", ModelVersionStatus.State.AVAILABLE),
        create_model_status_response(2, ErrorCode.UNKNOWN, "UNKNOWN",
                                     ModelVersionStatus.State.LOADING)])

    expected_output = {
        1: {
            "state": "AVAILABLE",
            "error_code": ErrorCode.OK,
            "error_message": "OK"
        },
        2: {
            "state": "LOADING",
            "error_code": ErrorCode.UNKNOWN,
            "error_message": "UNKNOWN"
        },
    }
    valid_grpc_serving_client_min.model_service_stub.GetModelStatus\
        = mocker.Mock(return_value=raw_response)

    response = valid_grpc_serving_client_min.get_model_status("model_name")

    assert valid_grpc_serving_client_min.model_service_stub.GetModelStatus.call_count == 1
    assert response == expected_output


@pytest.mark.parametrize("grpc_error_status_code, grpc_error_details,"
                         "raised_error_type, raised_error_message", COMMON_INVALID_GRPC)
def test_get_model_status_invalid_grpc(mocker, valid_grpc_serving_client_min,
                                       grpc_error_status_code, grpc_error_details,
                                       raised_error_type, raised_error_message):

    valid_grpc_serving_client_min.model_service_stub.GetModelStatus\
        = mocker.Mock(side_effect=create_grpc_error(grpc_error_status_code, grpc_error_details))

    with pytest.raises(raised_error_type) as grpc_error:
        valid_grpc_serving_client_min.get_model_status("model_name")

    assert str(grpc_error.value) == raised_error_message
    assert valid_grpc_serving_client_min.model_service_stub.GetModelStatus.call_count == 1


@pytest.mark.parametrize("model_raw_metadata_response_dict", MODEL_METADATA_RESPONSE_VALID)
def test_get_model_status_malformed_response(mocker, valid_grpc_serving_client_min,
                                             model_raw_metadata_response_dict):
    # Using metadata response to simulate situation with bad response returning from the server
    raw_response = create_model_metadata_response(model_raw_metadata_response_dict)

    valid_grpc_serving_client_min.model_service_stub.GetModelStatus\
        = mocker.Mock(return_value=raw_response)

    with pytest.raises(BadResponseError):
        valid_grpc_serving_client_min.get_model_status("model_name")

    assert valid_grpc_serving_client_min.model_service_stub.GetModelStatus.call_count == 1


@pytest.mark.parametrize("params, expected_error, error_message", MODEL_METADATA_INVALID_PARAMS)
def test_get_model_metadata_invalid_params(mocker, valid_grpc_serving_client_min,
                                           params, expected_error, error_message):
    valid_grpc_serving_client_min.prediction_service_stub.GetModelMetadata\
        = mocker.Mock()

    with pytest.raises(expected_error) as error:
        valid_grpc_serving_client_min.get_model_metadata(*params)

    valid_grpc_serving_client_min.prediction_service_stub.GetModelMetadata.call_count == 0
    assert str(error.value) == error_message


def test_get_model_metadata_valid(mocker, valid_grpc_serving_client_min,
                                  valid_model_metadata_response):
    expected_output = {'inputs': {'0': {'dtype': 'DT_FLOAT', 'shape': [1, 3, 244, 244]}},
                       'model_version': 2,
                       'outputs': {'1463': {'dtype': 'DT_FLOAT', 'shape': [1, 1000]}}}

    valid_grpc_serving_client_min.prediction_service_stub.GetModelMetadata\
        = mocker.Mock(return_value=valid_model_metadata_response.raw_response)

    response = valid_grpc_serving_client_min.get_model_metadata("model_name")

    assert valid_grpc_serving_client_min.prediction_service_stub.GetModelMetadata.call_count == 1
    assert response == expected_output


@pytest.mark.parametrize("grpc_error_status_code, grpc_error_details,"
                         "raised_error_type, raised_error_message", COMMON_INVALID_GRPC)
def test_get_model_metadata_invalid_grpc(mocker, valid_grpc_serving_client_min,
                                         grpc_error_status_code, grpc_error_details,
                                         raised_error_type, raised_error_message):

    valid_grpc_serving_client_min.prediction_service_stub.GetModelMetadata\
        = mocker.Mock(side_effect=create_grpc_error(grpc_error_status_code, grpc_error_details))

    with pytest.raises(raised_error_type) as grpc_error:
        valid_grpc_serving_client_min.get_model_metadata("model_name")

    assert str(grpc_error.value) == raised_error_message
    assert valid_grpc_serving_client_min.prediction_service_stub.GetModelMetadata.call_count == 1


def test_get_model_metadata_malformed_response(mocker, valid_grpc_serving_client_min,):
    # Using status response to simulate situation with bad response returning from the server
    raw_response = create_model_status_response(1, 0, "OK", ModelVersionStatus.State.AVAILABLE)

    valid_grpc_serving_client_min.prediction_service_stub.GetModelMetadata\
        = mocker.Mock(return_value=raw_response)

    with pytest.raises(BadResponseError):
        valid_grpc_serving_client_min.get_model_metadata("model_name")

    assert valid_grpc_serving_client_min.prediction_service_stub.GetModelMetadata.call_count == 1


@pytest.mark.parametrize("params, expected_error, error_message", PREDICT_INVALID_PARAMS)
def test_predict_invalid_params(mocker, valid_grpc_serving_client_min,
                                params, expected_error, error_message):

    valid_grpc_serving_client_min.prediction_service_stub.Predict\
        = mocker.Mock()

    with pytest.raises(expected_error) as error:
        valid_grpc_serving_client_min.predict(*params)

    valid_grpc_serving_client_min.prediction_service_stub.Predict.call_count == 0
    assert str(error.value) == error_message


@pytest.mark.parametrize("outputs_dict, model_name, model_version,"
                         "expected_outputs", PREDICT_RESPONSE_VALID)
def test_predict_valid(mocker, valid_grpc_serving_client_min,
                       outputs_dict, model_name, model_version,
                       expected_outputs):

    predict_response = create_predict_response(outputs_dict)
    valid_grpc_serving_client_min.prediction_service_stub.Predict\
        = mocker.Mock(return_value=predict_response.raw_response)

    mock_inputs = {"input": [1, 2, 3]}
    response = valid_grpc_serving_client_min.predict(mock_inputs, model_name, model_version)

    assert valid_grpc_serving_client_min.prediction_service_stub.Predict.call_count == 1

    if isinstance(response, dict):
        for output_name, array in response.items():
            assert output_name in predict_response.raw_response.outputs.keys()
            assert type(array) is np.ndarray
            assert np.array_equal(array, expected_outputs[output_name])
    else:
        assert type(response) is np.ndarray
        assert np.array_equal(response, expected_outputs)


@pytest.mark.parametrize("grpc_error_status_code, grpc_error_details,"
                         "raised_error_type, raised_error_message", PREDICT_INVALID_GRPC)
def test_predict_invalid_grpc(mocker, valid_grpc_serving_client_min,
                              grpc_error_status_code, grpc_error_details,
                              raised_error_type, raised_error_message):

    valid_grpc_serving_client_min.prediction_service_stub.Predict\
        = mocker.Mock(side_effect=create_grpc_error(grpc_error_status_code, grpc_error_details))

    mock_inputs = {"input": [1, 2, 3]}
    with pytest.raises(raised_error_type) as grpc_error:
        valid_grpc_serving_client_min.predict(mock_inputs, "model_name")

    assert str(grpc_error.value) == raised_error_message
    assert valid_grpc_serving_client_min.prediction_service_stub.Predict.call_count == 1


@pytest.mark.parametrize("outputs_dict, model_name, model_version, _,"
                         "expected_message", PREDICT_RESPONSE_TENSOR_TYPE_INVALID)
def test_predict_invalid_dtype_returned(mocker, valid_grpc_serving_client_min,
                                        outputs_dict, model_name, model_version,
                                        _, expected_message):

    predict_response = create_predict_response(outputs_dict)
    valid_grpc_serving_client_min.prediction_service_stub.Predict\
        = mocker.Mock(return_value=predict_response.raw_response)

    mock_inputs = {"input": [1, 2, 3]}
    with pytest.raises(BadResponseError) as parsing_error:
        valid_grpc_serving_client_min.predict(mock_inputs, model_name, model_version)

    assert valid_grpc_serving_client_min.prediction_service_stub.Predict.call_count == 1
    assert expected_message in str(parsing_error)
