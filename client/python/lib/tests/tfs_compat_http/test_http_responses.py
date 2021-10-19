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
import numpy as np

from collections import namedtuple
from numpy.core.numeric import array_equal

from ovmsclient.tfs_compat.http.responses import (HttpPredictResponse)

from tfs_compat_http.config import (PREDICT_RESPONSE_VALID_OUTPUTS,
                                    PREDICT_RESPONSE_VALID_OTHER)


def outputsEqual(outputs, expected_outputs):
    outputs = outputs["outputs"]
    expected_outputs = expected_outputs["outputs"]
    if isinstance(outputs, dict):
        for key in outputs:
            return array_equal(outputs[key], expected_outputs[key])
    elif isinstance(outputs, np.ndarray):
        return array_equal(outputs, expected_outputs)


RawResponseMock = namedtuple("RawResponse", "text")


@pytest.mark.parametrize("response, expected_output", PREDICT_RESPONSE_VALID_OUTPUTS)
def test_PredictResponse_to_dict_valid_outputs(response, expected_output):
    predict_raw_response = RawResponseMock(response)

    predict_response = HttpPredictResponse(predict_raw_response)
    output = predict_response.to_dict()

    assert(outputsEqual(output, expected_output))


@pytest.mark.parametrize("response, expected_output", PREDICT_RESPONSE_VALID_OTHER)
def test_PredictResponse_to_dict_valid_other(response, expected_output):
    predict_raw_response = RawResponseMock(response)

    predict_response = HttpPredictResponse(predict_raw_response)
    output = predict_response.to_dict()

    assert(output == expected_output)
