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
from tensorflow.core.protobuf.error_codes_pb2 import Code
from tensorflow.core.framework.types_pb2 import DataType

from ovmsclient.tfs_compat.grpc.serving_client import (_check_address, _open_certificate,
                                                       _check_config,
                                                       _check_port, _open_private_key,
                                                       _check_tls_config, _prepare_certs,
                                                       make_grpc_client,
                                                       _check_model_status_request,
                                                       _check_model_metadata_request,
                                                       _check_predict_request)
from ovmsclient.tfs_compat.grpc.requests import GrpcModelStatusRequest
from ovmsclient.tfs_compat.grpc.responses import GrpcModelStatusResponse
from ovmsclient.tfs_compat.grpc.requests import GrpcModelMetadataRequest
from ovmsclient.tfs_compat.grpc.responses import GrpcModelMetadataResponse
from ovmsclient.tfs_compat.grpc.responses import GrpcPredictResponse

from config import ADDRESS_INVALID, ADDRESS_VALID
from config import PORT_VALID, PORT_INVALID
from config import TLS_CONFIG_VALID, TLS_CONFIG_INVALID
from config import CONFIG_INVALID, CONFIG_VALID
from config import CERTIFICATE_VALID
from config import PRIVATE_KEY_VALID
from config import CHANNEL_CERTS_VALID
from config import BUILD_INVALID_CONFIG, BUILD_VALID, BUILD_INVALID_CERTS
from config import (MODEL_STATUS_REQUEST_VALID, MODEL_STATUS_REQUEST_INVALID_RAW_REQUEST,
                    MODEL_STATUS_REQUEST_INVALID_REQUEST_TYPE)
from config import GET_MODEL_STATUS_INVALID_GRPC
from config import (MODEL_METADATA_REQUEST_VALID, MODEL_METADATA_REQUEST_INVALID_RAW_REQUEST,
                    MODEL_METADATA_REQUEST_INVALID_REQUEST_TYPE)
from config import GET_MODEL_METADATA_INVALID_GRPC
from config import (PREDICT_REQUEST_VALID_SPEC, PREDICT_REQUEST_INVALID_SPEC_RAW_REQUEST,
                    PREDICT_REQUEST_INVALID_SPEC_TYPE)
from config import PREDICT_INVAlID_GRPC

from test_grpc_responses import create_model_metadata_response
from test_grpc_responses import create_model_status_response, merge_model_status_responses


@pytest.fixture
def valid_grpc_serving_client_min():
    config = {
        "address": "localhost",
        "port": 9000
    }
    return make_grpc_client(config)


@pytest.fixture
def valid_model_status_response():
    raw_response = merge_model_status_responses([
        create_model_status_response(1, Code.OK, "OK", ModelVersionStatus.State.AVAILABLE),
        create_model_status_response(0, Code.UNKNOWN, "UNKNOWN", ModelVersionStatus.State.LOADING)
    ])
    return GrpcModelStatusResponse(raw_response)


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


@pytest.mark.parametrize("address", ADDRESS_VALID)
def test_check_address_valid(address):
    _check_address(address)


@pytest.mark.parametrize("address, expected_exception, expected_message", ADDRESS_INVALID)
def test_check_address_invalid(address, expected_exception, expected_message):
    with pytest.raises(expected_exception) as e_info:
        _check_address(address)

    assert str(e_info.value) == expected_message


@pytest.mark.parametrize("address", PORT_VALID)
def test_check_port_valid(address):
    _check_port(address)


@pytest.mark.parametrize("address, expected_exception, expected_message", PORT_INVALID)
def test_check_port_invalid(address, expected_exception, expected_message):
    with pytest.raises(expected_exception) as e_info:
        _check_port(address)

    assert str(e_info.value) == expected_message


@pytest.mark.parametrize("tls_config, isfile_called_count", TLS_CONFIG_VALID)
def test_check_tls_config_valid(mocker, tls_config, isfile_called_count):
    mock_method = mocker.patch('os.path.isfile')
    _check_tls_config(tls_config)

    assert mock_method.call_count == isfile_called_count


@pytest.mark.parametrize("tls_config, expected_exception, expected_message,"
                         "isfile_called_count, is_valid_path", TLS_CONFIG_INVALID)
def test_check_tls_config_invalid(mocker, tls_config, expected_exception,
                                  expected_message, isfile_called_count, is_valid_path):
    mock_method = mocker.patch('os.path.isfile', side_effect=is_valid_path)
    with pytest.raises(expected_exception) as e_info:
        _check_tls_config(tls_config)

    assert str(e_info.value) == expected_message
    assert mock_method.call_count == isfile_called_count


@pytest.mark.parametrize("config, method_call_count", CONFIG_VALID)
def test_check_config_valid(mocker, config, method_call_count):
    mock_check_address = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_address')
    mock_check_port = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_port')
    mock_check_tls_config = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                         '._check_tls_config')

    _check_config(config)

    assert mock_check_address.call_count == method_call_count['check_address']
    assert mock_check_port.call_count == method_call_count['check_port']
    assert mock_check_tls_config.call_count == method_call_count['check_tls_config']


@pytest.mark.parametrize("config, method_call_count, expected_exception,"
                         "expected_message, side_effect", CONFIG_INVALID)
def test_check_config_invalid(mocker, config, method_call_count,
                              expected_exception, expected_message, side_effect):
    mock_check_address = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_address')
    mock_check_address.side_effect = side_effect['check_address']

    mock_check_port = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_port')
    mock_check_port.side_effect = side_effect['check_port']

    mock_check_tls_config = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                         '._check_tls_config')
    mock_check_tls_config.side_effect = side_effect['check_tls_config']

    with pytest.raises(expected_exception) as e_info:
        _check_config(config)

    assert str(e_info.value) == expected_message
    assert mock_check_address.call_count == method_call_count['check_address']
    assert mock_check_port.call_count == method_call_count['check_port']
    assert mock_check_tls_config.call_count == method_call_count['check_tls_config']


@pytest.mark.parametrize("certificate_path", CERTIFICATE_VALID)
def test_open_certificate_valid(mocker, certificate_path):
    mock_open = mocker.patch('builtins.open', mocker.mock_open(read_data='certificate'))

    certificate = _open_certificate(certificate_path)

    assert mock_open.call_count == 1
    assert certificate == 'certificate'


@pytest.mark.parametrize("private_key_path", PRIVATE_KEY_VALID)
def test_open_private_key_valid(mocker, private_key_path):
    mock_open = mocker.patch('builtins.open', mocker.mock_open(read_data='privatekey'))

    private_key = _open_private_key(private_key_path)

    assert mock_open.call_count == 1
    assert private_key == 'privatekey'


@pytest.mark.parametrize("server_cert, client_cert, client_key,"
                         "method_call_count", CHANNEL_CERTS_VALID)
def test_prepare_certs_valid(mocker, server_cert, client_cert, client_key, method_call_count):
    mock_open_certificate = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                         '._open_certificate')
    mock_check_key_valid = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                        '._open_private_key')

    _prepare_certs(server_cert, client_cert, client_key)

    assert mock_open_certificate.call_count == method_call_count['check_certificate_valid']
    assert mock_check_key_valid.call_count == method_call_count['check_key_valid']


@pytest.mark.parametrize("config, method_call_count", BUILD_VALID)
def test_make_grpc_client_valid(mocker, config, method_call_count):
    mock_check_config = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_config')
    mock_prepare_certs = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '._prepare_certs',
                                      return_value=(b'server_certificate',
                                                    b'client_certificate',
                                                    b'client_key'))

    client = make_grpc_client(config)

    assert mock_check_config.call_count == method_call_count['check_config']
    assert mock_prepare_certs.call_count == method_call_count['prepare_certs']
    assert type(client.channel) == Channel
    assert type(client.model_service_stub) == ModelServiceStub
    assert type(client.prediction_service_stub) == PredictionServiceStub


@pytest.mark.parametrize("config, expected_exception, expected_message,"
                         "method_call_count", BUILD_INVALID_CONFIG)
def test_make_grpc_client_invalid_config(mocker, config, expected_exception,
                                         expected_message, method_call_count):
    mock_check_config = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                     '._check_config',
                                     side_effect=expected_exception(expected_message))
    mock_prepare_certs = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._prepare_certs')

    with pytest.raises(expected_exception) as e_info:
        make_grpc_client(config)

    assert str(e_info.value) == expected_message
    assert mock_check_config.call_count == method_call_count['check_config']
    assert mock_prepare_certs.call_count == method_call_count['prepare_certs']


@pytest.mark.parametrize("config, expected_exception, expected_message,"
                         "method_call_count", BUILD_INVALID_CERTS)
def test_make_grpc_client_invalid_certs(mocker, config, expected_exception,
                                        expected_message, method_call_count):
    mock_check_config = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_config')
    mock_prepare_certs = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '._prepare_certs',
                                      side_effect=expected_exception(expected_message))

    with pytest.raises(expected_exception) as e_info:
        make_grpc_client(config)

    assert str(e_info.value) == expected_message
    assert mock_check_config.call_count == method_call_count['check_config']
    assert mock_prepare_certs.call_count == method_call_count['prepare_certs']


def test_get_model_status_valid(mocker, valid_grpc_serving_client_min,
                                valid_model_status_response):
    model_status_request = mocker.Mock()

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '._check_model_status_request')
    valid_grpc_serving_client_min.model_service_stub.GetModelStatus\
        = mocker.Mock(return_value=valid_model_status_response.raw_response)

    response = valid_grpc_serving_client_min.get_model_status(model_status_request)

    assert mock_check_request.call_count == 1
    assert valid_grpc_serving_client_min.model_service_stub.GetModelStatus.call_count == 1
    assert type(response) == type(valid_model_status_response)
    assert response.raw_response == valid_model_status_response.raw_response


@pytest.mark.parametrize("expected_message, grpc_error_status_code,"
                         "grpc_error_details", GET_MODEL_STATUS_INVALID_GRPC)
def test_get_model_status_invalid_grpc(mocker, valid_grpc_serving_client_min,
                                       expected_message, grpc_error_status_code,
                                       grpc_error_details):
    model_status_request = mocker.Mock()

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '._check_model_status_request')
    valid_grpc_serving_client_min.model_service_stub.GetModelStatus\
        = mocker.Mock(side_effect=create_grpc_error(grpc_error_status_code, grpc_error_details))

    with pytest.raises(ConnectionError) as e_info:
        valid_grpc_serving_client_min.get_model_status(model_status_request)

    assert str(e_info.value) == expected_message
    assert mock_check_request.call_count == 1
    assert valid_grpc_serving_client_min.model_service_stub.GetModelStatus.call_count == 1


@pytest.mark.parametrize("request_parameters_dict, expected_exception,"
                         "expected_message", MODEL_STATUS_REQUEST_INVALID_RAW_REQUEST)
def test_get_model_status_invalid_raw_request(mocker, valid_grpc_serving_client_min,
                                              request_parameters_dict, expected_exception,
                                              expected_message):
    model_status_request\
        = create_model_status_request(request_parameters_dict['model_name'],
                                      request_parameters_dict['model_version'],
                                      request_parameters_dict['raw_request_model_name'],
                                      request_parameters_dict['raw_request_model_version'])

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '._check_model_status_request',
                                      side_effect=expected_exception(expected_message))
    with pytest.raises(expected_exception) as e_info:
        valid_grpc_serving_client_min.get_model_status(model_status_request)

    assert str(e_info.value) == expected_message
    assert mock_check_request.call_count == 1


@pytest.mark.parametrize("model_status_request, expected_exception,"
                         "expected_message", MODEL_STATUS_REQUEST_INVALID_REQUEST_TYPE)
def test_get_model_status_invalid_request_type(mocker, valid_grpc_serving_client_min,
                                               model_status_request, expected_exception,
                                               expected_message):
    mock_check_request = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '._check_model_status_request',
                                      side_effect=expected_exception(expected_message))
    with pytest.raises(expected_exception) as e_info:
        valid_grpc_serving_client_min.get_model_status(model_status_request)

    assert str(e_info.value) == expected_message
    assert mock_check_request.call_count == 1


@pytest.mark.parametrize("request_parameters_dict", MODEL_STATUS_REQUEST_VALID)
def test_check_model_status_request_valid(request_parameters_dict):
    model_status_request\
        = create_model_status_request(request_parameters_dict['model_name'],
                                      request_parameters_dict['model_version'],
                                      request_parameters_dict['raw_request_model_name'],
                                      request_parameters_dict['raw_request_model_version'])

    _check_model_status_request(model_status_request)


@pytest.mark.parametrize("request_parameters_dict, expected_exception,"
                         "expected_message", MODEL_STATUS_REQUEST_INVALID_RAW_REQUEST)
def test_check_model_status_request_invalid_raw_request(request_parameters_dict, expected_exception,
                                                        expected_message):
    model_status_request\
        = create_model_status_request(request_parameters_dict['model_name'],
                                      request_parameters_dict['model_version'],
                                      request_parameters_dict['raw_request_model_name'],
                                      request_parameters_dict['raw_request_model_version'])

    with pytest.raises(expected_exception) as e_info:
        _check_model_status_request(model_status_request)

    assert str(e_info.value) == expected_message


@pytest.mark.parametrize("model_status_request, expected_exception,"
                         "expected_message", MODEL_STATUS_REQUEST_INVALID_REQUEST_TYPE)
def test_check_model_status_request_invalid_type(model_status_request, expected_exception,
                                                 expected_message):
    with pytest.raises(expected_exception) as e_info:
        _check_model_status_request(model_status_request)

    assert str(e_info.value) == expected_message


@pytest.mark.parametrize("request_parameters_dict", MODEL_METADATA_REQUEST_VALID)
def test_check_model_metadata_request_valid(request_parameters_dict):
    model_metadata_request\
        = create_model_metadata_request(request_parameters_dict['model_name'],
                                        request_parameters_dict['model_version'],
                                        request_parameters_dict['raw_request_model_name'],
                                        request_parameters_dict['raw_request_model_version'],
                                        request_parameters_dict['metadata_field_list'])

    _check_model_metadata_request(model_metadata_request)


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
        _check_model_metadata_request(model_metadata_request)

    assert str(e_info.value) == expected_message


@pytest.mark.parametrize("model_metadata_request, expected_exception,"
                         "expected_message", MODEL_METADATA_REQUEST_INVALID_REQUEST_TYPE)
def test_check_model_metadata_request_invalid_type(model_metadata_request,
                                                   expected_exception, expected_message):
    with pytest.raises(expected_exception) as e_info:
        _check_model_metadata_request(model_metadata_request)

    assert str(e_info.value) == expected_message


def test_get_model_metadata_valid(mocker, valid_grpc_serving_client_min,
                                  valid_model_metadata_response):
    model_metadata_request = mocker.Mock()

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '._check_model_metadata_request')
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
                                      '._check_model_metadata_request')
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
                                      '._check_model_metadata_request',
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
                                      '._check_model_metadata_request',
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
    _check_predict_request(predict_request)


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
        _check_predict_request(predict_request)

    assert str(e_info.value) == expected_message


@pytest.mark.parametrize("predict_request, expected_exception,"
                         "expected_message", PREDICT_REQUEST_INVALID_SPEC_TYPE)
def test_check_predict_request_invalid_type(predict_request, expected_exception, expected_message):
    with pytest.raises(expected_exception) as e_info:
        _check_predict_request(predict_request)

    assert str(e_info.value) == expected_message


def test_predict_valid(mocker, valid_grpc_serving_client_min, valid_predict_response):
    predict_request = mocker.Mock()

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client'
                                      '._check_predict_request')
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
                                      '._check_predict_request')
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
                                      '._check_predict_request',
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
                                      '._check_predict_request',
                                      side_effect=expected_exception(expected_message))

    with pytest.raises(expected_exception) as e_info:
        valid_grpc_serving_client_min.predict(predict_request)

    assert str(e_info.value) == expected_message
    assert mock_check_request.call_count == 1
