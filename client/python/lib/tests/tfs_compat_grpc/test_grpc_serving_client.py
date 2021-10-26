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
from numpy import float64, array, int32
from ovmsclient.tfs_compat.grpc.requests import GrpcPredictRequest

from tensorflow_serving.apis.get_model_metadata_pb2 import GetModelMetadataRequest
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from tensorflow_serving.apis.model_service_pb2_grpc import ModelServiceStub
from tensorflow_serving.apis.predict_pb2 import PredictRequest, PredictResponse
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusRequest
from tensorflow_serving.apis.get_model_status_pb2 import ModelVersionStatus
from tensorflow.core.protobuf.error_codes_pb2 import Code as ErrorCode
from tensorflow.core.framework.types_pb2 import DataType

from ovmsclient.tfs_compat.grpc.serving_client import (make_grpc_client,
                                                       GrpcClient)
from ovmsclient.tfs_compat.grpc.requests import GrpcModelStatusRequest
from ovmsclient.tfs_compat.grpc.responses import GrpcModelStatusResponse
from ovmsclient.tfs_compat.grpc.requests import GrpcModelMetadataRequest
from ovmsclient.tfs_compat.grpc.responses import GrpcModelMetadataResponse
from ovmsclient.tfs_compat.grpc.responses import GrpcPredictResponse

from tfs_compat_grpc.config import (BUILD_INVALID_CONFIG, BUILD_VALID, BUILD_INVALID_CERTS,
                                    MODEL_STATUS_REQUEST_VALID,
                                    MODEL_STATUS_REQUEST_INVALID_RAW_REQUEST,
                                    MODEL_STATUS_REQUEST_INVALID_REQUEST_TYPE,
                                    GET_MODEL_STATUS_INVALID_GRPC,
                                    MODEL_METADATA_REQUEST_VALID,
                                    MODEL_METADATA_REQUEST_INVALID_RAW_REQUEST,
                                    MODEL_METADATA_REQUEST_INVALID_REQUEST_TYPE,
                                    GET_MODEL_METADATA_INVALID_GRPC,
                                    PREDICT_REQUEST_VALID_SPEC,
                                    PREDICT_REQUEST_INVALID_SPEC_RAW_REQUEST,
                                    PREDICT_REQUEST_INVALID_SPEC_TYPE,
                                    PREDICT_INVAlID_GRPC)

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


@pytest.fixture
def valid_predict_response():
    predict_raw_response = PredictResponse()

    predict_raw_response.model_spec.name = 'name'
    predict_raw_response.model_spec.version.value = 0

    outputs_dict = {
        "0": TensorProto(dtype=DataType.DT_INT32,
                         tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=2),
                                                       TensorShapeProto.Dim(size=3)]),
                         tensor_content=array([1, 2, 3, 4, 5, 6], dtype=int32).tobytes()),
        "1": TensorProto(dtype=DataType.DT_DOUBLE,
                         tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
                         double_val=array([12.0], dtype=float64)),
    }

    for key, value in outputs_dict.items():
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

@pytest.mark.parametrize("params, expected_error, error_message", [
    # Model name check
    ([("model", "name"), 1, 10], TypeError, "model_name type should be string, but is tuple"),
    # Model version check
    (["model_name", "model_version", 10], TypeError, "model_version type should be int, but is str"),
    (["model_name", 2**63, 10], ValueError, f"model_version should be in range <0, {2**63-1}>"),
    (["model_name", -1, 10], ValueError, f"model_version should be in range <0, {2**63-1}>"),
    # Timeout check
    (["model_name", 1, "string"], TypeError, "timeout value must be positive float"),
    (["model_name", 1, 0], TypeError, "timeout value must be positive float"),
    (["model_name", 1, -1], TypeError, "timeout value must be positive float"),
])

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
        create_model_status_response(2, ErrorCode.UNKNOWN, "UNKNOWN", ModelVersionStatus.State.LOADING)])

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
                         "raised_error_type, raised_error_message", GET_MODEL_STATUS_INVALID_GRPC)
def test_get_model_status_invalid_grpc(mocker, valid_grpc_serving_client_min,
                                       grpc_error_status_code, grpc_error_details, 
                                       raised_error_type, raised_error_message):

    valid_grpc_serving_client_min.model_service_stub.GetModelStatus\
        = mocker.Mock(side_effect=create_grpc_error(grpc_error_status_code, grpc_error_details))

    with pytest.raises(raised_error_type) as grpc_error:
        valid_grpc_serving_client_min.get_model_status("model_name")

    assert str(grpc_error.value) == raised_error_message
    assert valid_grpc_serving_client_min.model_service_stub.GetModelStatus.call_count == 1


@pytest.mark.parametrize("request_parameters_dict", MODEL_METADATA_REQUEST_VALID)
def test_check_model_metadata_request_valid(request_parameters_dict):
    model_metadata_request\
        = create_model_metadata_request(request_parameters_dict['model_name'],
                                        request_parameters_dict['model_version'],
                                        request_parameters_dict['raw_request_model_name'],
                                        request_parameters_dict['raw_request_model_version'],
                                        request_parameters_dict['metadata_field_list'])

    GrpcClient._check_model_metadata_request(model_metadata_request)


@pytest.mark.parametrize("request_parameters_dict, expected_exception,"
                         "expected_message", MODEL_METADATA_REQUEST_INVALID_RAW_REQUEST)
def test_check_model_metadata_request_invalid_raw_request(request_parameters_dict,
                                                          expected_exception,
                                                          expected_message):
    model_metadata_request\
        = create_model_metadata_request(request_parameters_dict['model_name'],
                                        request_parameters_dict['model_version'],
                                        request_parameters_dict['raw_request_model_name'],
                                        request_parameters_dict['raw_request_model_version'],
                                        request_parameters_dict['metadata_field_list'])

    with pytest.raises(expected_exception) as e_info:
        GrpcClient._check_model_metadata_request(model_metadata_request)

    assert str(e_info.value) == expected_message


@pytest.mark.parametrize("model_metadata_request, expected_exception,"
                         "expected_message", MODEL_METADATA_REQUEST_INVALID_REQUEST_TYPE)
def test_check_model_metadata_request_invalid_type(model_metadata_request,
                                                   expected_exception, expected_message):
    with pytest.raises(expected_exception) as e_info:
        GrpcClient._check_model_metadata_request(model_metadata_request)

    assert str(e_info.value) == expected_message


def test_get_model_metadata_valid(mocker, valid_grpc_serving_client_min,
                                  valid_model_metadata_response):
    model_metadata_request = mocker.Mock()

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '.GrpcClient._check_model_metadata_request')
    valid_grpc_serving_client_min.prediction_service_stub.GetModelMetadata\
        = mocker.Mock(return_value=valid_model_metadata_response.raw_response)

    response = valid_grpc_serving_client_min.get_model_metadata(model_metadata_request)

    assert mock_check_request.call_count == 1
    assert valid_grpc_serving_client_min.prediction_service_stub.GetModelMetadata.call_count == 1
    assert type(response) == type(valid_model_metadata_response)
    assert response.raw_response == valid_model_metadata_response.raw_response


@pytest.mark.parametrize("expected_message, grpc_error_status_code,"
                         "grpc_error_details", GET_MODEL_METADATA_INVALID_GRPC)
def test_get_model_metadata_invalid_grpc(mocker, valid_grpc_serving_client_min,
                                         expected_message, grpc_error_status_code,
                                         grpc_error_details):
    model_metadata_request = mocker.Mock()

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '.GrpcClient._check_model_metadata_request')
    valid_grpc_serving_client_min.prediction_service_stub.GetModelMetadata\
        = mocker.Mock(side_effect=create_grpc_error(grpc_error_status_code, grpc_error_details))

    with pytest.raises(ConnectionError) as e_info:
        valid_grpc_serving_client_min.get_model_metadata(model_metadata_request)

    assert str(e_info.value) == expected_message
    assert mock_check_request.call_count == 1
    assert valid_grpc_serving_client_min.prediction_service_stub.GetModelMetadata.call_count == 1


@pytest.mark.parametrize("request_parameters_dict, expected_exception,"
                         "expected_message", MODEL_METADATA_REQUEST_INVALID_RAW_REQUEST)
def test_get_model_metadata_invalid_raw_request(mocker, valid_grpc_serving_client_min,
                                                request_parameters_dict, expected_exception,
                                                expected_message):
    model_metadata_request \
        = create_model_metadata_request(request_parameters_dict['model_name'],
                                        request_parameters_dict['model_version'],
                                        request_parameters_dict['raw_request_model_name'],
                                        request_parameters_dict['raw_request_model_version'],
                                        request_parameters_dict['metadata_field_list'])

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '.GrpcClient._check_model_metadata_request',
                                      side_effect=expected_exception(expected_message))
    with pytest.raises(expected_exception) as e_info:
        valid_grpc_serving_client_min.get_model_metadata(model_metadata_request)

    assert str(e_info.value) == expected_message
    assert mock_check_request.call_count == 1


@pytest.mark.parametrize("model_metadata_request, expected_exception,"
                         "expected_message", MODEL_METADATA_REQUEST_INVALID_REQUEST_TYPE)
def test_get_model_metadata_invalid_type(mocker, valid_grpc_serving_client_min,
                                         model_metadata_request, expected_exception,
                                         expected_message):
    mock_check_request = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '.GrpcClient._check_model_metadata_request',
                                      side_effect=expected_exception(expected_message))

    with pytest.raises(expected_exception) as e_info:
        valid_grpc_serving_client_min.get_model_metadata(model_metadata_request)

    assert str(e_info.value) == expected_message
    assert mock_check_request.call_count == 1


@pytest.mark.parametrize("request_parameter_dict", PREDICT_REQUEST_VALID_SPEC)
def test_check_predict_request_valid(request_parameter_dict):
    predict_request = create_predict_request(request_parameter_dict['model_name'],
                                             request_parameter_dict['model_version'],
                                             request_parameter_dict['raw_request_model_name'],
                                             request_parameter_dict['raw_request_model_version'],
                                             request_parameter_dict['inputs_dict'],
                                             request_parameter_dict['raw_request_inputs_dict'])
    GrpcClient._check_predict_request(predict_request)


@pytest.mark.parametrize("request_parameter_dict,"
                         "expected_exception, expected_message",
                         PREDICT_REQUEST_INVALID_SPEC_RAW_REQUEST)
def test_check_predict_request_invalid_raw_request(request_parameter_dict, expected_exception,
                                                   expected_message):
    predict_request = create_predict_request(request_parameter_dict['model_name'],
                                             request_parameter_dict['model_version'],
                                             request_parameter_dict['raw_request_model_name'],
                                             request_parameter_dict['raw_request_model_version'],
                                             request_parameter_dict['inputs_dict'],
                                             request_parameter_dict['raw_request_inputs_dict'])
    with pytest.raises(expected_exception) as e_info:
        GrpcClient._check_predict_request(predict_request)

    assert str(e_info.value) == expected_message


@pytest.mark.parametrize("predict_request, expected_exception,"
                         "expected_message", PREDICT_REQUEST_INVALID_SPEC_TYPE)
def test_check_predict_request_invalid_type(predict_request, expected_exception, expected_message):
    with pytest.raises(expected_exception) as e_info:
        GrpcClient._check_predict_request(predict_request)

    assert str(e_info.value) == expected_message


def test_predict_valid(mocker, valid_grpc_serving_client_min, valid_predict_response):
    predict_request = mocker.Mock()

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '.GrpcClient._check_predict_request')
    valid_grpc_serving_client_min.prediction_service_stub.Predict\
        = mocker.Mock(return_value=valid_predict_response.raw_response)

    response = valid_grpc_serving_client_min.predict(predict_request)

    assert mock_check_request.call_count == 1
    assert valid_grpc_serving_client_min.prediction_service_stub.Predict.call_count == 1
    assert type(response) == type(valid_predict_response)
    assert response.raw_response == valid_predict_response.raw_response


@pytest.mark.parametrize("expected_message, grpc_error_status_code,"
                         "grpc_error_details", PREDICT_INVAlID_GRPC)
def test_predict_invalid_grpc(mocker, valid_grpc_serving_client_min,
                              expected_message, grpc_error_status_code,
                              grpc_error_details):
    predict_request = mocker.Mock()

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '.GrpcClient._check_predict_request')
    valid_grpc_serving_client_min.prediction_service_stub.Predict\
        = mocker.Mock(side_effect=create_grpc_error(grpc_error_status_code, grpc_error_details))

    with pytest.raises(ConnectionError) as e_info:
        valid_grpc_serving_client_min.predict(predict_request)

    assert str(e_info.value) == expected_message
    assert mock_check_request.call_count == 1
    assert valid_grpc_serving_client_min.prediction_service_stub.Predict.call_count == 1


@pytest.mark.parametrize("request_parameter_dict, expected_exception,"
                         "expected_message", PREDICT_REQUEST_INVALID_SPEC_RAW_REQUEST)
def test_predict_invalid_raw_request(mocker, valid_grpc_serving_client_min,
                                     request_parameter_dict, expected_exception,
                                     expected_message):
    predict_request = create_predict_request(request_parameter_dict['model_name'],
                                             request_parameter_dict['model_version'],
                                             request_parameter_dict['raw_request_model_name'],
                                             request_parameter_dict['raw_request_model_version'],
                                             request_parameter_dict['inputs_dict'],
                                             request_parameter_dict['raw_request_inputs_dict'])
    mock_check_request = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '.GrpcClient._check_predict_request',
                                      side_effect=expected_exception(expected_message))

    with pytest.raises(expected_exception) as e_info:
        valid_grpc_serving_client_min.predict(predict_request)

    assert str(e_info.value) == expected_message
    assert mock_check_request.call_count == 1


@pytest.mark.parametrize("predict_request, expected_exception,"
                         "expected_message", PREDICT_REQUEST_INVALID_SPEC_TYPE)
def test_predict_invalid_type(mocker, valid_grpc_serving_client_min,
                              predict_request, expected_exception, expected_message):
    mock_check_request = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '.GrpcClient._check_predict_request',
                                      side_effect=expected_exception(expected_message))

    with pytest.raises(expected_exception) as e_info:
        valid_grpc_serving_client_min.predict(predict_request)

    assert str(e_info.value) == expected_message
    assert mock_check_request.call_count == 1
