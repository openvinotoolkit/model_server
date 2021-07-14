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

from ovmsclient.tfs_compat.grpc.serving_client import _check_port
from config import PORT_VALID, PORT_INVALID

@pytest.mark.parametrize("address", PORT_VALID)
def test_check_address_valid(address):
    _check_port(address)

@pytest.mark.parametrize("address, expected_exception, expected_message", PORT_INVALID)
def test_check_address_invalid(address, expected_exception, expected_message):
    with pytest.raises(expected_exception) as e_info:
        _check_port(address)
        assert str(e_info.value) == expected_message
