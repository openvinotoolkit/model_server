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
import numpy as np

from ovmsclient.tfs_compat.http.serving_client import HttpClient, make_http_client
from ovmsclient.tfs_compat.http.responses import HttpModelMetadataResponse
from ovmsclient.tfs_compat.http.requests import HttpModelMetadataRequest
from ovmsclient.tfs_compat.base.errors import BadResponseError

from config import (MODEL_STATUS_INVALID_PARAMS, PREDICT_INVALID_PARAMS)

from tfs_compat_http.config import (BUILD_VALID, BUILD_INVALID_CONFIG, COMMON_RESPONSE_ERROR,
                                    GET_MODEL_STATUS_VALID,
                                    METADATA_RESPONSE_VALID_OUTPUTS,
                                    MODEL_METADATA_REQUEST_INVALID_REQUEST_TYPE,
                                    PREDICT_RESPONSE_ERROR,
                                    PREDICT_RESPONSE_MALFROMED_RESPONSE,
                                    PREDICT_RESPONSE_VALID_OUTPUTS,
                                    STATUS_RESPONSE_MALFROMED_RESPONSE, RawResponseMock)


@pytest.fixture
def valid_http_serving_client_min():
    return make_http_client("localhost:9000")


# Remove and use RawResponseMock from config after refactoring all endpoints
def mocked_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self, text):
            self.text = text

    return MockResponse(args)
# -----


@pytest.mark.parametrize("config, method_call_count, expected_client_key, expected_server_cert",
                         BUILD_VALID)
def test_make_http_client_valid(mocker, config, method_call_count, expected_client_key,
                                expected_server_cert):
    mock_check_url = mocker.patch('ovmsclient.tfs_compat.base.serving_client'
                                  '.ServingClient._check_url')
    mock_check_tls_config = mocker.patch('ovmsclient.tfs_compat.base.serving_client'
                                         '.ServingClient._check_tls_config')
    client = make_http_client(**config)

    assert mock_check_url.call_count == method_call_count['_check_url']
    assert mock_check_tls_config.call_count == method_call_count['_check_tls_config']
    assert client.url == config['url']
    assert client.client_key == expected_client_key
    assert client.server_cert == expected_server_cert
    assert isinstance(client.session, requests.Session)


@pytest.mark.parametrize("config, method_call_spec, expected_exception, expected_message",
                         BUILD_INVALID_CONFIG)
def test_make_http_client_invalid_config(mocker, config, method_call_spec, expected_exception,
                                         expected_message):
    mocks = []
    for method_name, call_spec in method_call_spec.items():
        call_count, error_raised = call_spec
        mock = mocker.patch(f"ovmsclient.tfs_compat.base.serving_client."
                            f"ServingClient.{method_name}", side_effect=error_raised)
        mocks.append((mock, call_count))

    with pytest.raises(expected_exception) as e_info:
        make_http_client(**config)

    assert str(e_info.value) == expected_message
    for mock_info in mocks:
        mock, call_count = mock_info
        assert mock.call_count == call_count


@pytest.mark.parametrize("response, expected_output", GET_MODEL_STATUS_VALID)
def test_get_model_status_valid(mocker, valid_http_serving_client_min,
                                response, expected_output):
    raw_response = RawResponseMock(*response)
    valid_http_serving_client_min.session.get\
        = mocker.Mock(return_value=raw_response)

    response = valid_http_serving_client_min.get_model_status("model_name")

    assert valid_http_serving_client_min.session.get.call_count == 1
    assert response == expected_output


@pytest.mark.parametrize("source_error, raised_error", [
                         (requests.exceptions.ConnectionError, ConnectionError),
                         (requests.exceptions.ReadTimeout, TimeoutError)
                         ])
def test_get_model_status_connection_error(mocker, valid_http_serving_client_min,
                                           source_error, raised_error):

    valid_http_serving_client_min.session.get\
        = mocker.Mock(side_effect=source_error())

    with pytest.raises(raised_error):
        valid_http_serving_client_min.get_model_status("model_name")

    assert valid_http_serving_client_min.session.get.call_count == 1


@pytest.mark.parametrize("params, expected_error, error_message", MODEL_STATUS_INVALID_PARAMS)
def test_get_model_status_invalid_params(mocker, valid_http_serving_client_min,
                                         params, expected_error, error_message):

    valid_http_serving_client_min.session.get\
        = mocker.Mock()

    with pytest.raises(expected_error) as error:
        valid_http_serving_client_min.get_model_status(*params)

    assert valid_http_serving_client_min.session.get.call_count == 0
    assert str(error.value) == error_message


@pytest.mark.parametrize("response, expected_error, expected_message", COMMON_RESPONSE_ERROR)
def test_get_model_status_server_error(mocker, valid_http_serving_client_min,
                                       response, expected_error, expected_message):

    raw_response = RawResponseMock(*response)
    valid_http_serving_client_min.session.get\
        = mocker.Mock(return_value=raw_response)

    with pytest.raises(expected_error) as error:
        valid_http_serving_client_min.get_model_status("model")

    assert valid_http_serving_client_min.session.get.call_count == 1
    assert str(error.value) == expected_message


@pytest.mark.parametrize("response, _", STATUS_RESPONSE_MALFROMED_RESPONSE)
def test_get_model_status_malformed_response(mocker, valid_http_serving_client_min,
                                             response, _):

    raw_response = RawResponseMock(*response)
    valid_http_serving_client_min.session.get\
        = mocker.Mock(return_value=raw_response)

    with pytest.raises(BadResponseError):
        valid_http_serving_client_min.get_model_status("model_name")

    assert valid_http_serving_client_min.session.get.call_count == 1


@pytest.mark.parametrize("response, expected_output", METADATA_RESPONSE_VALID_OUTPUTS)
def test_get_model_metadata_valid(mocker, valid_http_serving_client_min,
                                response, expected_output):
    raw_response = RawResponseMock(*response)
    valid_http_serving_client_min.session.get\
        = mocker.Mock(return_value=raw_response)

    response = valid_http_serving_client_min.get_model_metadata("model_name")

    assert valid_http_serving_client_min.session.get.call_count == 1
    assert response == expected_output

@pytest.mark.parametrize("params, expected_error, error_message", [
    # Model name check
    ([("model", "name"), 1, 10], TypeError, "model_name type should be string, but is tuple"),
    # Model version check
    (["model_name", "model_version", 10], TypeError,
        "model_version type should be int, but is str"),
    (["model_name", 2**63, 10], ValueError, f"model_version should be in range <0, {2**63-1}>"),
    (["model_name", -1, 10], ValueError, f"model_version should be in range <0, {2**63-1}>"),
    # Timeout check
    (["model_name", 1, "string"], TypeError, "timeout value must be positive float"),
    (["model_name", 1, 0], TypeError, "timeout value must be positive float"),
    (["model_name", 1, -1], TypeError, "timeout value must be positive float"),
])
def test_get_model_metadata_invalid_params(mocker, valid_http_serving_client_min,
                                         params, expected_error, error_message):

    valid_http_serving_client_min.session.get\
        = mocker.Mock()

    with pytest.raises(expected_error) as error:
        valid_http_serving_client_min.get_model_metadata(*params)

    assert valid_http_serving_client_min.session.get.call_count == 0
    assert str(error.value) == error_message


@pytest.mark.parametrize("response, expected_error, expected_message", COMMON_RESPONSE_ERROR)
def test_get_model_metadata_server_error(mocker, valid_http_serving_client_min,
                                       response, expected_error, expected_message):

    raw_response = RawResponseMock(*response)
    valid_http_serving_client_min.session.get\
        = mocker.Mock(return_value=raw_response)

    with pytest.raises(expected_error) as error:
        valid_http_serving_client_min.get_model_metadata("model_name")

    assert valid_http_serving_client_min.session.get.call_count == 1
    assert str(error.value) == expected_message


def test_check_model_metadata_request_valid():
    HttpClient._check_model_metadata_request(HttpModelMetadataRequest("model_name", 0))


@pytest.mark.parametrize("model_metadata_request, expected_exception,"
                         "expected_message", MODEL_METADATA_REQUEST_INVALID_REQUEST_TYPE)
def test_check_model_metadata_request_invalid_type(model_metadata_request, expected_exception,
                                                   expected_message):
    with pytest.raises(expected_exception) as e_info:
        HttpClient._check_model_metadata_request(model_metadata_request)

    assert str(e_info.value) == expected_message


@pytest.mark.parametrize("response, expected_output", PREDICT_RESPONSE_VALID_OUTPUTS)
def test_predict_valid(mocker, valid_http_serving_client_min, response, expected_output):
    raw_response = RawResponseMock(*response)
    valid_http_serving_client_min.session.post\
        = mocker.Mock(return_value=raw_response)

    mocked_inputs = {"input": [1, 2, 3]}
    response = valid_http_serving_client_min.predict(mocked_inputs, "model_name")

    assert valid_http_serving_client_min.session.post.call_count == 1
    if isinstance(response, dict):
        for key in response:
            assert np.array_equal(response[key], expected_output[key])
    elif isinstance(response, np.ndarray):
        assert np.array_equal(response, expected_output)


@pytest.mark.parametrize("source_error, raised_error", [
                         (requests.exceptions.ConnectionError, ConnectionError),
                         (requests.exceptions.ReadTimeout, TimeoutError)
                         ])
def test_predict_connection_error(mocker, valid_http_serving_client_min,
                                  source_error, raised_error):

    valid_http_serving_client_min.session.post\
        = mocker.Mock(side_effect=source_error())

    mocked_inputs = {"input": [1, 2, 3]}
    with pytest.raises(raised_error):
        valid_http_serving_client_min.predict(mocked_inputs, "model_name")

    assert valid_http_serving_client_min.session.post.call_count == 1


@pytest.mark.parametrize("params, expected_error, error_message", PREDICT_INVALID_PARAMS)
def test_predict_invalid_params(mocker, valid_http_serving_client_min,
                                params, expected_error, error_message):

    valid_http_serving_client_min.session.post\
        = mocker.Mock()

    with pytest.raises(expected_error) as error:
        valid_http_serving_client_min.predict(*params)

    assert valid_http_serving_client_min.session.post.call_count == 0
    assert str(error.value) == error_message


@pytest.mark.parametrize("response, expected_error, expected_message", PREDICT_RESPONSE_ERROR)
def test_predict_server_error(mocker, valid_http_serving_client_min,
                              response, expected_error, expected_message):

    raw_response = RawResponseMock(*response)
    valid_http_serving_client_min.session.post\
        = mocker.Mock(return_value=raw_response)

    mocked_inputs = {"input": [1, 2, 3]}
    with pytest.raises(expected_error) as error:
        valid_http_serving_client_min.predict(mocked_inputs, "model_name")

    assert valid_http_serving_client_min.session.post.call_count == 1
    assert str(error.value) == expected_message


@pytest.mark.parametrize("response, _", PREDICT_RESPONSE_MALFROMED_RESPONSE)
def test_predict_malformed_response(mocker, valid_http_serving_client_min,
                                    response, _):

    raw_response = RawResponseMock(*response)
    valid_http_serving_client_min.session.post\
        = mocker.Mock(return_value=raw_response)

    mocked_inputs = {"input": [1, 2, 3]}
    with pytest.raises(BadResponseError):
        valid_http_serving_client_min.predict(mocked_inputs, "model_name")

    assert valid_http_serving_client_min.session.post.call_count == 1
