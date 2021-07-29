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

from grpc._channel import Channel
import pytest

from tensorflow_serving.apis.model_service_pb2_grpc import ModelServiceStub
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub

from ovmsclient.tfs_compat.grpc.serving_client import (_check_address, _open_certificate, _check_config, _check_port,
_open_private_key, _check_tls_config, _prepare_certs, make_grpc_client)

from config import ADDRESS_INVALID, ADDRESS_VALID
from config import PORT_VALID, PORT_INVALID
from config import TLS_CONFIG_VALID, TLS_CONFIG_INVALID
from config import CONFIG_INVALID, CONFIG_VALID
from config import CERTIFICATE_VALID
from config import PRIVATE_KEY_VALID
from config import CHANNEL_CERTS_VALID
from config import BUILD_INVALID_CONFIG, BUILD_VALID, BUILD_INVALID_CERTS

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
    assert mock_method.call_count ==  isfile_called_count

@pytest.mark.parametrize("tls_config, expected_exception, expected_message, isfile_called_count, is_valid_path", TLS_CONFIG_INVALID)
def test_check_tls_config_invalid(mocker, tls_config, expected_exception, expected_message, isfile_called_count, is_valid_path):
    mock_method = mocker.patch('os.path.isfile', side_effect=is_valid_path)
    with pytest.raises(expected_exception) as e_info:
        _check_tls_config(tls_config)
        assert str(e_info.value) == expected_message
    
    assert mock_method.call_count ==  isfile_called_count

@pytest.mark.parametrize("config, method_call_count", CONFIG_VALID)
def test_check_config_valid(mocker, config, method_call_count):
    mock_check_address = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_address')
    mock_check_port = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_port')
    mock_check_tls_config = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_tls_config')
    
    _check_config(config)

    assert mock_check_address.call_count == method_call_count['check_address']
    assert mock_check_port.call_count == method_call_count['check_port']
    assert mock_check_tls_config.call_count == method_call_count['check_tls_config']

@pytest.mark.parametrize("config, method_call_count, expected_exception, expected_message, side_effect", CONFIG_INVALID)
def test_check_config_invalid(mocker, config, method_call_count, expected_exception, expected_message, side_effect):
    mock_check_address = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_address')
    mock_check_address.side_effect = side_effect['check_address']

    mock_check_port = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_port')
    mock_check_port.side_effect = side_effect['check_port']

    mock_check_tls_config = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_tls_config')
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

@pytest.mark.parametrize("server_cert, client_cert, client_key, method_call_count", CHANNEL_CERTS_VALID)
def test_prepare_certs_valid(mocker, server_cert, client_cert, client_key, method_call_count):
    mock_open_certificate = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._open_certificate')
    mock_check_key_valid = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._open_private_key')

    _prepare_certs(server_cert, client_cert, client_key)

    assert mock_open_certificate.call_count == method_call_count['check_certificate_valid']
    assert mock_check_key_valid.call_count == method_call_count['check_key_valid']

@pytest.mark.parametrize("config, method_call_count", BUILD_VALID)
def test_make_grpc_client_valid(mocker, config, method_call_count):
    mock_check_config = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_config')
    mock_prepare_certs = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._prepare_certs', return_value=(b'server_certificate', b'client_certificate', b'client_key'))

    client = make_grpc_client(config)

    assert mock_check_config.call_count == method_call_count['check_config']
    assert mock_prepare_certs.call_count == method_call_count['prepare_certs']
    assert type(client.channel) == Channel
    assert type(client.model_service_stub) == ModelServiceStub
    assert type(client.prediction_service_stub) == PredictionServiceStub

@pytest.mark.parametrize("config, expected_exception, expected_message, method_call_count", BUILD_INVALID_CONFIG)
def test_make_grpc_client_invalid_config(mocker, config, expected_exception, expected_message, method_call_count):
    mock_check_config = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_config', side_effect=expected_exception(expected_message))
    mock_prepare_certs = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._prepare_certs')

    with pytest.raises(expected_exception) as e_info:
        client = make_grpc_client(config)
        assert str(e_info.value) == expected_message

    assert mock_check_config.call_count == method_call_count['check_config']
    assert mock_prepare_certs.call_count == method_call_count['prepare_certs']

@pytest.mark.parametrize("config, expected_exception, expected_message, method_call_count", BUILD_INVALID_CERTS)
def test_make_grpc_client_invalid_certs(mocker, config, expected_exception, expected_message, method_call_count):
    mock_check_config = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_config')
    mock_prepare_certs = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._prepare_certs', side_effect=expected_exception(expected_message))

    with pytest.raises(expected_exception) as e_info:
        client = make_grpc_client(config)
        assert str(e_info.value) == expected_message

    assert mock_check_config.call_count == method_call_count['check_config']
    assert mock_prepare_certs.call_count == method_call_count['prepare_certs']
