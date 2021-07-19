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

from ovmsclient.tfs_compat.grpc.serving_client import _check_certificate_valid
from config import CERTIFICATE_INVALID, CERTIFICATE_VALID

@pytest.mark.parametrize("certificate_path", CERTIFICATE_VALID)
def test_check_certificate_valid_valid(mocker, certificate_path):
    mock_load_certificate = mocker.patch('OpenSSL.crypto.load_certificate')

    mock_open = mocker.patch('builtins.open', mocker.mock_open(read_data='certificate'))

    certificate = _check_certificate_valid(certificate_path)
    assert mock_load_certificate.call_count == 1
    assert mock_open.call_count == 1
    assert certificate == 'certificate'
    

@pytest.mark.parametrize("certificate_path, expected_exception, expected_message, load_certificate_side_effect", CERTIFICATE_INVALID)
def test_check_certificate_valid_invalid(mocker, certificate_path, expected_exception, expected_message, load_certificate_side_effect):
    mock_load_certificate = mocker.patch('OpenSSL.crypto.load_certificate', side_effect=load_certificate_side_effect)

    mock_open = mocker.patch('builtins.open')

    with pytest.raises(expected_exception) as e_info:
        _check_certificate_valid(certificate_path)
        assert str(e_info.value) == expected_message

    assert mock_load_certificate.call_count == 1
    assert mock_open.call_count == 1
