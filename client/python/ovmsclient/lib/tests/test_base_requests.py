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

from ovmsclient.tfs_compat.base.requests import _check_model_spec

from config import (MODEL_SPEC_INVALID, MODEL_SPEC_VALID)


@pytest.mark.parametrize("name, version", MODEL_SPEC_VALID)
def test_check_model_spec_valid(name, version):
    _check_model_spec(name, version)


@pytest.mark.parametrize("name, version, expected_exception, expected_message", MODEL_SPEC_INVALID)
def test_check_model_spec_invalid(name, version, expected_exception, expected_message):
    with pytest.raises(expected_exception) as e_info:
        _check_model_spec(name, version)

    assert str(e_info.value) == expected_message
