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

from ovmsclient.tfs_compat.grpc.serving_client import  _check_tls_config
from config import TLS_CONFIG_VALID, TLS_CONFIG_INVALID

@pytest.mark.parametrize("tls_config", TLS_CONFIG_VALID)
def test_check_address_valid(mocker, tls_config):
    mock_path = mocker.Mock(name="path")
    mock_path.get.return_value = is_valid_path
    mocker.patch('os.path.isfile', new=mock_path)
    _check_tls_config(tls_config)

@pytest.mark.parametrize("tls_config, expected_exception, expected_message", TLS_CONFIG_INVALID)
def test_check_address_invalid(mocker, tls_config, expected_exception, expected_message):
    mock_path = mocker.Mock(name="path")
    mock_path.get.return_value = is_valid_path
    mocker.patch('os.path.isfile', new=mock_path)
    with pytest.raises(expected_exception) as e_info:
        _check_tls_config(tls_config)
        assert str(e_info.value) == expected_message

def is_valid_path(path):
    if path == 'valid_path':
        return True
    return False