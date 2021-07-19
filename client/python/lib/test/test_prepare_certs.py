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

from ovmsclient.tfs_compat.grpc.serving_client import _prepare_certs
from config import CHANNEL_CERTS_INVALID, CHANNEL_CERTS_VALID

@pytest.mark.parametrize("server_cert, client_cert, client_key, method_call_count", CHANNEL_CERTS_VALID)
def test_prepare_certs_valid(mocker, server_cert, client_cert, client_key, method_call_count):
    mock_check_certificate_valid = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_certificate_valid')
    mock_check_key_valid = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_private_key_valid')

    _prepare_certs(server_cert, client_cert, client_key)

    assert mock_check_certificate_valid.call_count == method_call_count['check_certificate_valid']
    assert mock_check_key_valid.call_count == method_call_count['check_key_valid']

@pytest.mark.parametrize("server_cert, client_cert, client_key, method_call_count, expected_exception, expected_message, side_effect", CHANNEL_CERTS_INVALID)
def test_prepare_certs_invalid(mocker, server_cert, client_cert, client_key, method_call_count, expected_exception, expected_message, side_effect):
    mock_check_certificate_valid = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_certificate_valid', side_effect=side_effect["check_certificate_valid"])
    mock_check_key_valid = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_private_key_valid', side_effect=side_effect["check_key_valid"])

    with pytest.raises(expected_exception) as e_info:
        _prepare_certs(server_cert, client_cert, client_key)
        assert str(e_info.value) == expected_message

    assert mock_check_certificate_valid.call_count == method_call_count['check_certificate_valid']
    assert mock_check_key_valid.call_count == method_call_count['check_key_valid']
