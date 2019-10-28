import pytest
import numpy as np
from config import JSON_CHECK_TEST_CASES

from ie_serving.server.constants import INVALID_FORMAT, COLUMN_FORMAT


def test_predict_wrong_model(client):
    response = client.simulate_request(method='POST',
                                       path='/v1/models/fake_model:predict',
                                       headers={
                                           "Content-Type": "application/json"},
                                       body=None)
    assert response.status_code == 404


def test_predict_wrong_model_version(client):
    response = client.simulate_request(method='POST',
                                       path='/v1/models/test/versions/5'
                                            ':predict',
                                       headers={
                                           "Content-Type": "application/json"},
                                       body=None)
    assert response.status_code == 404


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


def test_predict_malformed_input_data(mocker, client):
    body = "{\"valid\": \"json\"}"
    get_input_format_mock = mocker.patch(
        'ie_serving.server.rest_service.get_input_format')
    preprocess_json_request_mock = mocker.patch(
        'ie_serving.server.rest_service.preprocess_json_request')
    prepare_input_data_mock = mocker.patch(
        'ie_serving.server.rest_service.prepare_input_data')
    results_mock = mocker.patch(
        'ie_serving.server.request.Request.wait_for_result')

    get_input_format_mock.return_value = COLUMN_FORMAT
    preprocess_json_request_mock.return_value = {"input": []}
    prepare_input_data_mock.return_value = {"input": np.ones(shape=(
        1, 1, 1))}, None
    results_mock.return_value = "Malformed input", 1

    response = client.simulate_request(method='POST',
                                       path='/v1/models/test:predict',
                                       headers={
                                           "Content-Type": "application/json"},
                                       body=body)
    assert get_input_format_mock.called
    assert preprocess_json_request_mock.called
    assert prepare_input_data_mock.called
    assert results_mock.called

    assert "Malformed input" in response.text
    assert response.status_code == 400


def test_predict_successful(mocker, client):
    body = "{\"valid\": \"json\"}"
    get_input_format_mock = mocker.patch(
        'ie_serving.server.rest_service.get_input_format')
    preprocess_json_request_mock = mocker.patch(
        'ie_serving.server.rest_service.preprocess_json_request')
    prepare_input_data_mock = mocker.patch(
        'ie_serving.server.rest_service.prepare_input_data')
    results_mock = mocker.patch(
        'ie_serving.server.request.Request.wait_for_result')
    prepare_json_response_mock = mocker.patch(
        'ie_serving.server.rest_service.prepare_json_response')

    get_input_format_mock.return_value = COLUMN_FORMAT
    prepare_input_data_mock.return_value = {"input": np.ones(shape=(
        1, 1, 1))}, None

    results_mock.return_value = {"outputs": np.ones(shape=(1, 1))}, 1
    prepare_json_response_mock.return_value = {"outputs": [1, 1]}

    response = client.simulate_request(method='POST',
                                       path='/v1/models/test:predict',
                                       headers={
                                           "Content-Type": "application/json"},
                                       body=body)
    assert get_input_format_mock.called
    assert preprocess_json_request_mock.called
    assert prepare_input_data_mock.called
    assert results_mock.called
    assert prepare_json_response_mock.called

    assert "{\"outputs\": [1, 1]}" in response.text
    assert response.status_code == 200
