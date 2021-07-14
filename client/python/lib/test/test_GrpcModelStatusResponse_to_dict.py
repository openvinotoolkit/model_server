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

@pytest.mark.parametrize("model_raw_response_list" , RAW_MODEL_RESPONSE_VALID)
def test_GrpcModelStatusResponse_to_dict_valid(model_raw_response_list):
    model_raw_responses = []
    for model_raw_response in model_raw_response_list:
        model_raw_responses.append(create_model_status_response(model_raw_response[0], model_raw_response[1], model_raw_response[2], model_raw_response[3]))
    raw_response = merge_model_status_responses(model_raw_responses)

    response = GrpcModelStatusResponse(raw_response)
    response_dictionary = response.to_dict()
    assert isinstance(response_dictionary, dict)
    assert len(response_dictionary) == len(model_raw_response_list)
    for model_raw_response in model_raw_response_list:
        assert model_raw_response[0] in response_dictionary
        assert isinstance(response_dictionary[model_raw_response[0]], dict)
        assert response_dictionary[model_raw_response[0]]['error_code'] == model_raw_response[1]
        assert response_dictionary[model_raw_response[0]]['error_message'] == model_raw_response[2]
        assert response_dictionary[model_raw_response[0]]['state'] == GrpcModelStatusResponse._STATE_TO_STRING_MAPPING[model_raw_response[3]]

