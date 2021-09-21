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
import requests

from ovmsclient.tfs_compat.http.serving_client import make_http_client

from tfs_compat_http.config import (BUILD_VALID, BUILD_INVALID_CONFIG)


@pytest.mark.parametrize("config, method_call_count, expected_client_key, expected_server_cert",
                         BUILD_VALID)
def test_make_http_client_valid(mocker, config, method_call_count, expected_client_key,
                                expected_server_cert):
    mock_check_config = mocker.patch('ovmsclient.tfs_compat.base.serving_client'
                                     '.ServingClient._check_config')

    client = make_http_client(config)

    assert mock_check_config.call_count == method_call_count['check_config']
    assert client.address == config['address']
    assert client.port == config['port']
    assert client.client_key == expected_client_key
    assert client.server_cert == expected_server_cert
    assert isinstance(client.session, requests.Session)


@pytest.mark.parametrize("config, expected_exception, expected_message,"
                         "method_call_count", BUILD_INVALID_CONFIG)
def test_make_http_client_invalid_config(mocker, config, expected_exception,
                                         expected_message, method_call_count):
    mock_check_config = mocker.patch('ovmsclient.tfs_compat.base.serving_client'
                                     '.ServingClient._check_config',
                                     side_effect=expected_exception(expected_message))

    with pytest.raises(expected_exception) as e_info:
        make_http_client(config)

    assert str(e_info.value) == expected_message
    assert mock_check_config.call_count == method_call_count['check_config']
