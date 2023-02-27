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


from ovmsclient.tfs_compat.http.requests import (HttpPredictRequest, HttpModelMetadataRequest,
                                                 HttpModelStatusRequest, make_predict_request,
                                                 _parse_input_data, make_metadata_request,
                                                 make_status_request)

from tfs_compat_http.config import (PREDICT_REQUEST_INVALID_INPUTS,
                                    PREDICT_REQUEST_VALID,
                                    PARSE_INPUT_DATA_VALID,
                                    PARSE_INPUT_DATA_INVALID)

from config import (MODEL_SPEC_INVALID)


@pytest.mark.parametrize("inputs, expected_parsed_inputs, name, version", PREDICT_REQUEST_VALID)
def test_make_predict_request_valid(inputs, expected_parsed_inputs, name, version):
    model_predict_request = make_predict_request(inputs, name, version)

    parsed_inputs = model_predict_request.parsed_inputs

    assert isinstance(model_predict_request, HttpPredictRequest)
    assert model_predict_request.model_name == name
    assert model_predict_request.model_version == version
    assert model_predict_request.inputs == inputs
    assert isinstance(parsed_inputs, str)
    assert parsed_inputs == expected_parsed_inputs


@pytest.mark.parametrize("name, version, expected_exception, expected_message", MODEL_SPEC_INVALID)
def test_make_predict_request_invalid_model_spec(mocker, name, version,
                                                 expected_exception, expected_message):
    inputs = {
        "input": [1, 2, 3]
    }
    mock_method = mocker.patch('ovmsclient.tfs_compat.http.requests._check_model_spec',
                               side_effect=expected_exception(expected_message))
    with pytest.raises(expected_exception) as e_info:
        make_predict_request(inputs, name, version)

    assert str(e_info.value) == expected_message
    mock_method.assert_called_once()


@pytest.mark.causes_deprecation_warning(triggered_by="numpy<1.24")
@pytest.mark.parametrize("""inputs, name, version,
                            expected_exception, expected_message""", PREDICT_REQUEST_INVALID_INPUTS)
def test_make_predict_request_invalid_inputs(mocker, inputs, name, version,
                                             expected_exception, expected_message):
    mock_method = mocker.patch('ovmsclient.tfs_compat.http.requests._check_model_spec')
    with pytest.raises(expected_exception) as e_info:
        make_predict_request(inputs, name, version)

    assert str(e_info.value) == expected_message
    mock_method.assert_called_once()


@pytest.mark.parametrize("input, expected_parsed_input", PARSE_INPUT_DATA_VALID)
def test_parse_input_data_valid(input, expected_parsed_input):
    parsed_input = _parse_input_data(input)
    assert parsed_input == expected_parsed_input


@pytest.mark.causes_deprecation_warning(triggered_by="numpy<1.24")
@pytest.mark.parametrize("""input, expected_exception, expected_message""",
                         PARSE_INPUT_DATA_INVALID)
def test_parse_input_data_invalid(input, expected_exception, expected_message):
    with pytest.raises(expected_exception) as e_info:
        _parse_input_data(input)
    assert str(e_info.value) == expected_message


@pytest.mark.parametrize("name, version, expected_exception, expected_message", MODEL_SPEC_INVALID)
def test_make_metadata_request_invalid_model_spec(mocker, name, version,
                                                  expected_exception, expected_message):
    mock_method = mocker.patch('ovmsclient.tfs_compat.http.requests._check_model_spec',
                               side_effect=expected_exception(expected_message))
    with pytest.raises(expected_exception) as e_info:
        make_metadata_request(name, version)

    assert str(e_info.value) == expected_message
    mock_method.assert_called_once()


def test_make_metadata_request_valid():
    model_name = "model"
    model_version = 3
    model_metadata_request = make_metadata_request(model_name, model_version)
    assert isinstance(model_metadata_request, HttpModelMetadataRequest)
    assert model_metadata_request.model_name == model_name
    assert model_metadata_request.model_version == model_version


@pytest.mark.parametrize("name, version, expected_exception, expected_message", MODEL_SPEC_INVALID)
def test_make_status_request_invalid_model_spec(mocker, name, version,
                                                expected_exception, expected_message):
    mock_method = mocker.patch('ovmsclient.tfs_compat.http.requests._check_model_spec',
                               side_effect=expected_exception(expected_message))
    with pytest.raises(expected_exception) as e_info:
        make_status_request(name, version)

    assert str(e_info.value) == expected_message
    mock_method.assert_called_once()


def test_make_status_request_valid():
    model_name = "model"
    model_version = 3
    model_status_request = make_status_request(model_name, model_version)
    assert isinstance(model_status_request, HttpModelStatusRequest)
    assert model_status_request.model_name == model_name
    assert model_status_request.model_version == model_version
