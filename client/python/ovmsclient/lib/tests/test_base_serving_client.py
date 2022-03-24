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


import pytest
from ovmsclient.tfs_compat.base.serving_client import ServingClient

from config import (TLS_CONFIG_VALID, TLS_CONFIG_INVALID,
                    URL_VALID, URL_INVALID,
                    CERTIFICATE_VALID,
                    PRIVATE_KEY_VALID,
                    CHANNEL_CERTS_VALID)


@pytest.mark.parametrize("tls_config, isfile_called_count", TLS_CONFIG_VALID)
def test_check_tls_config_valid(mocker, tls_config, isfile_called_count):
    mock_method = mocker.patch('os.path.isfile')
    ServingClient._check_tls_config(tls_config)

    assert mock_method.call_count == isfile_called_count


@pytest.mark.parametrize("tls_config, expected_exception, expected_message,"
                         "isfile_called_count, is_valid_path", TLS_CONFIG_INVALID)
def test_check_tls_config_invalid(mocker, tls_config, expected_exception,
                                  expected_message, isfile_called_count, is_valid_path):
    mock_method = mocker.patch('os.path.isfile', side_effect=is_valid_path)
    with pytest.raises(expected_exception) as e_info:
        ServingClient._check_tls_config(tls_config)

    assert str(e_info.value) == expected_message
    assert mock_method.call_count == isfile_called_count


@pytest.mark.parametrize("url", URL_VALID)
def test_check_url_valid(mocker, url):
    ServingClient._check_url(url)


@pytest.mark.parametrize("url, expected_exception, expected_message", URL_INVALID)
def test_check_url_invalid(mocker, url, expected_exception, expected_message):
    with pytest.raises(expected_exception):
        ServingClient._check_url(url)


@pytest.mark.parametrize("certificate_path", CERTIFICATE_VALID)
def test_open_certificate_valid(mocker, certificate_path):
    mock_open = mocker.patch('builtins.open', mocker.mock_open(read_data='certificate'))

    certificate = ServingClient._open_certificate(certificate_path)

    assert mock_open.call_count == 1
    assert certificate == 'certificate'


@pytest.mark.parametrize("private_key_path", PRIVATE_KEY_VALID)
def test_open_private_key_valid(mocker, private_key_path):
    mock_open = mocker.patch('builtins.open', mocker.mock_open(read_data='privatekey'))

    private_key = ServingClient._open_private_key(private_key_path)

    assert mock_open.call_count == 1
    assert private_key == 'privatekey'


@pytest.mark.parametrize("server_cert, client_cert, client_key,"
                         "method_call_count", CHANNEL_CERTS_VALID)
def test_prepare_certs_valid(mocker, server_cert, client_cert, client_key, method_call_count):
    mock_open_certificate = mocker.patch('ovmsclient.tfs_compat.base.serving_client'
                                         '.ServingClient._open_certificate')
    mock_check_key_valid = mocker.patch('ovmsclient.tfs_compat.base.serving_client'
                                        '.ServingClient._open_private_key')

    ServingClient._prepare_certs(server_cert, client_cert, client_key)

    assert mock_open_certificate.call_count == method_call_count['check_certificate_valid']
    assert mock_check_key_valid.call_count == method_call_count['check_key_valid']
