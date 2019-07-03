import pytest
import falcon
import json


@pytest.mark.parametrize(
    "body",
    [
        "just_text", 1234, "[]"
        "{\"key\": \"value\"","\"key\": \"value\"}", "\"key\": \"value\""
        "{key: value}", "{1: \"value\"}", "{\"value\"}",
        "{\"key\": 1 \"key2\": \"value\"}",
        "{\"key\": \"value\", \"key\": \"value\"}",

    ]
)
def test_predict_invalid_json(mocker, client, body):
    model_availability_mock = mocker.patch(
        'ie_serving.server.service_utils'
        '.check_availability_of_requested_model')

    model_availability_mock.return_value = True, 1
    result = client.simulate_request(method='POST',
                                     path='/v1/models/model:predict',
                                     headers={}, data=body)
    assert "Invalid JSON" in result.text
