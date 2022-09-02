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

from numpy.core.numeric import array_equal

from ovmsclient.tfs_compat.http.responses import (HttpPredictResponse,
                                                  HttpModelMetadataResponse,
                                                  HttpModelStatusResponse)

from tfs_compat_http.config import (PREDICT_RESPONSE_VALID_OUTPUTS,
                                    METADATA_RESPONSE_VALID_OUTPUTS,
                                    STATUS_RESPONSE_VALID_OUTPUTS,
                                    STATUS_RESPONSE_MALFROMED_RESPONSE,
                                    COMMON_RESPONSE_ERROR,
                                    PREDICT_RESPONSE_MALFROMED_RESPONSE,
                                    PREDICT_RESPONSE_ERROR,
                                    RESPONSE_VALID_OTHER, RawResponseMock)


def outputsEqual(outputs, expected_outputs):
    outputs = outputs["outputs"]
    if isinstance(outputs, dict):
        for key in outputs:
            return array_equal(outputs[key], expected_outputs[key])
    elif isinstance(outputs, np.ndarray):
        return array_equal(outputs, expected_outputs)


@pytest.mark.parametrize("response, expected_output", PREDICT_RESPONSE_VALID_OUTPUTS)
def test_PredictResponse_to_dict_valid_outputs(response, expected_output):
    predict_raw_response = RawResponseMock(*response)

    predict_response = HttpPredictResponse(predict_raw_response)
    output = predict_response.to_dict()

    assert (outputsEqual(output, expected_output))


@pytest.mark.parametrize("response, expected_error", PREDICT_RESPONSE_MALFROMED_RESPONSE)
def test_PredictResponse_to_dict_malformed_response(response, expected_error):
    predict_raw_response = RawResponseMock(*response)

    predict_response = HttpPredictResponse(predict_raw_response)
    with pytest.raises(expected_error):
        predict_response.to_dict()


@pytest.mark.parametrize("response, expected_errors",
                         PREDICT_RESPONSE_ERROR)
def test_PredictResponse_to_dict_server_error(response, expected_errors):
    expected_error, expected_message = expected_errors["response_to_dict_error"]
    predict_raw_response = RawResponseMock(*response)

    predict_response = HttpPredictResponse(predict_raw_response)
    with pytest.raises(expected_error) as error:
        predict_response.to_dict()
    assert expected_message in str(error.value)


@pytest.mark.parametrize("response, expected_output", METADATA_RESPONSE_VALID_OUTPUTS)
def test_ModelMetadataResponse_to_dict_valid_outputs(response, expected_output):
    metadata_raw_response = RawResponseMock(*response)

    metadata_response = HttpModelMetadataResponse(metadata_raw_response)
    output = metadata_response.to_dict()

    assert (output == expected_output)


@pytest.mark.parametrize("response, expected_output", RESPONSE_VALID_OTHER)
def test_ModelMetadataResponse_to_dict_valid_other(response, expected_output):
    metadata_raw_response = RawResponseMock(*response)

    metadata_response = HttpModelMetadataResponse(metadata_raw_response)
    output = metadata_response.to_dict()

    assert (output == expected_output)


@pytest.mark.parametrize("response, expected_errors",
                         COMMON_RESPONSE_ERROR)
def test_ModelMetadataResponse_to_dict_common_server_error(response, expected_errors):
    expected_error, expected_message = expected_errors["response_to_dict_error"]
    metadata_raw_response = RawResponseMock(*response)

    metadata_response = HttpModelMetadataResponse(metadata_raw_response)
    with pytest.raises(expected_error) as error:
        metadata_response.to_dict()
    assert expected_message in str(error.value)


@pytest.mark.parametrize("response, expected_output", STATUS_RESPONSE_VALID_OUTPUTS)
def test_ModelStatusResponse_to_dict_valid_outputs(response, expected_output):
    status_raw_response = RawResponseMock(*response)

    status_response = HttpModelStatusResponse(status_raw_response)
    output = status_response.to_dict()

    assert (output == expected_output)


@pytest.mark.parametrize("response, expected_error", STATUS_RESPONSE_MALFROMED_RESPONSE)
def test_ModelStatusResponse_to_dict_malformed_response(response, expected_error):
    status_raw_response = RawResponseMock(*response)

    status_response = HttpModelStatusResponse(status_raw_response)
    with pytest.raises(expected_error):
        status_response.to_dict()


@pytest.mark.parametrize("response, expected_errors",
                         COMMON_RESPONSE_ERROR)
def test_ModelStatusResponse_to_dict_common_server_error(response, expected_errors):
    expected_error, expected_message = expected_errors["response_to_dict_error"]
    status_raw_response = RawResponseMock(*response)

    status_response = HttpModelStatusResponse(status_raw_response)
    with pytest.raises(expected_error) as error:
        status_response.to_dict()
    assert expected_message in str(error.value)
