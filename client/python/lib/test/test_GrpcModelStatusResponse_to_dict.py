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

from ovmsclient.tfs_compat.grpc.responses import GrpcModelStatusResponse
from config import MODEL_RESPONSE_CORRECT

@pytest.mark.parametrize("raw_response, expected" , MODEL_RESPONSE_CORRECT)
def test_GrpcModelStatusResponse_to_dict_correct(raw_response, expected):
    response = GrpcModelStatusResponse(raw_response)
    response_dictionary = response.to_dict()
    assert isinstance(response_dictionary, dict)
    assert len(response_dictionary) == expected
    for key in response_dictionary:
        assert len(response_dictionary[key]) == 3
        assert response_dictionary[key]['state'] == "START"
        assert response_dictionary[key]['error_code'] == 1
        assert response_dictionary[key]['error_message'] == "msg"
