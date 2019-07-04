import pytest

from config import JSON_CHECK_TEST_CASES


@pytest.mark.parametrize(
    "body", JSON_CHECK_TEST_CASES
)
def test_predict_invalid_json(client, body):
    result = client.simulate_request(method='POST',
                                     path='/v1/models/test:predict',
                                     headers={
                                         "Content-Type": "application/json"},
                                     body=body)
    assert "Invalid JSON" in result.text
