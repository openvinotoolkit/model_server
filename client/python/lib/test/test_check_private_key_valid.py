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

from ovmsclient.tfs_compat.grpc.serving_client import _check_private_key_valid
from config import PRIVATE_KEY_INVALID, PRIVATE_KEY_VALID

@pytest.mark.parametrize("private_key_path", PRIVATE_KEY_VALID)
def test_check_private_key_valid_valid(mocker, private_key_path):
    mock_load_private_key = mocker.patch('OpenSSL.crypto.load_privatekey')

    mock_open = mocker.patch('builtins.open', mocker.mock_open(read_data='privatekey'))

    private_key = _check_private_key_valid(private_key_path)
    assert mock_load_private_key.call_count == 1
    assert mock_open.call_count == 1
    assert private_key == 'privatekey'
    

@pytest.mark.parametrize("private_key_path, expected_exception, expected_message, load_privatekey_side_effect", PRIVATE_KEY_INVALID)
def test_check_private_key_valid_invalid(mocker, private_key_path, expected_exception, expected_message, load_privatekey_side_effect):
    mock_load_private_key = mocker.patch('OpenSSL.crypto.load_privatekey', side_effect=load_privatekey_side_effect)

    mock_open = mocker.patch('builtins.open')

    with pytest.raises(expected_exception) as e_info:
        _check_private_key_valid(private_key_path)
        assert str(e_info.value) == expected_message

    assert mock_load_private_key.call_count == 1
    assert mock_open.call_count == 1
