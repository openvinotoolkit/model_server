import pytest

from config import JSON_CHECK_TEST_CASES

from ie_serving.server.constants import INVALID_FORMAT


@pytest.mark.parametrize(
    "body", JSON_CHECK_TEST_CASES
)
def test_predict_invalid_json(client, body):
    response = client.simulate_request(method='POST',
                                       path='/v1/models/test:predict',
                                       headers={
                                           "Content-Type": "application/json"},
                                       body=body)
    assert "Invalid JSON" in response.text
    assert response.status_code == 400


def test_predict_invalid_input_format(mocker, client):
    body = "{\"valid\": \"json\"}"
    get_input_format_mock = mocker.patch(
        'ie_serving.server.rest_service.get_input_format')
    get_input_format_mock.return_value = INVALID_FORMAT
    response = client.simulate_request(method='POST',
                                       path='/v1/models/test:predict',
                                       headers={
                                           "Content-Type": "application/json"},
                                       body=body)
    assert get_input_format_mock.called
    assert "Invalid inputs" in response.text
    assert response.status_code == 400
