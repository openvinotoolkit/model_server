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
from config import RAW_MODEL_RESPONSE_VALID, create_model_status_response, merge_model_status_responses

@pytest.mark.parametrize("model_raw_response_dict" , RAW_MODEL_RESPONSE_VALID)
def test_GrpcModelStatusResponse_to_dict_valid(model_raw_response_dict):
    model_raw_responses = []
    for version in model_raw_response_dict:
        model_raw_responses.append(create_model_status_response(version, model_raw_response_dict[version][1], model_raw_response_dict[version][2],
         model_raw_response_dict[version][0]))
    raw_response = merge_model_status_responses(model_raw_responses)

    response = GrpcModelStatusResponse(raw_response)
    response_dictionary = response.to_dict()
    assert isinstance(response_dictionary, dict)
    assert len(response_dictionary) == len(model_raw_response_dict)
    for version in model_raw_response_dict:
        assert version in response_dictionary
        assert isinstance(response_dictionary[version], dict)
        assert response_dictionary[version]['error_code'] == model_raw_response_dict[version][1]
        assert response_dictionary[version]['error_message'] == model_raw_response_dict[version][2]
        assert response_dictionary[version]['state'] == GrpcModelStatusResponse._STATE_TO_STRING_MAPPING[model_raw_response_dict[version][0]]
