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

from ovmsclient.tfs_compat.grpc.serving_client import _check_config
from config import CONFIG_INVALID, CONFIG_VALID

@pytest.mark.parametrize("config, called_counts", CONFIG_VALID)
def test_check_config_valid(mocker, config, called_counts):
    mock_check_address = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_address')
    mock_check_port = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_port')
    mock_check_tls_config = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_tls_config')
    
    _check_config(config)

    assert mock_check_address.call_count == called_counts['check_address']
    assert mock_check_port.call_count == called_counts['check_port']
    assert mock_check_tls_config.call_count == called_counts['check_tls_config']

@pytest.mark.parametrize("config, called_counts, expected_exception, expected_message, side_effects", CONFIG_INVALID)
def test_check_config_invalid(mocker, config, called_counts, expected_exception, expected_message, side_effects):
    mock_check_address = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_address')
    if side_effects['check_address'] is not None:
        mock_check_address.side_effect = side_effects['check_address']

    mock_check_port = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_port')
    if side_effects['check_port'] is not None:
        mock_check_port.side_effect = side_effects['check_port']

    mock_check_tls_config = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_tls_config')
    if side_effects['check_tls_config'] is not None:
        mock_check_tls_config.side_effect = side_effects['check_tls_config']

    with pytest.raises(expected_exception) as e_info:
        _check_config(config)
        assert str(e_info.value) == expected_message

    assert mock_check_address.call_count == called_counts['check_address']
    assert mock_check_port.call_count == called_counts['check_port']
    assert mock_check_tls_config.call_count == called_counts['check_tls_config']

