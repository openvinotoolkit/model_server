import pytest
import falcon
import json

from ie_serving.server.constants import INVALID_FORMAT, COLUMN_FORMAT, \
    COLUMN_SIMPLIFIED, ROW_FORMAT, ROW_SIMPLIFIED
from ie_serving.server.rest_msg_validation import get_input_format, \
    _evaluate_inputs, _evaluate_instances
from tests.unit.config import INPUT_FORMAT_CHECK_TEST_CASES, \
    JSON_CHECK_TEST_CASES, \
    EVALUATE_INPUTS_TEST_CASES, EVALUATE_INSTANCES_TEST_CASES, \
    INPUTS_EVALUATION, INSTANCES_EVALUATION

# TO DO: Move to another file. Only tests for validation functions here.
from tests.unit.conftest import DEFAULT_INPUT_KEY


@pytest.mark.parametrize(
    "body", JSON_CHECK_TEST_CASES
)
def test_predict_invalid_json(client, body):
    result = client.simulate_request(method='POST',
                                     path='/v1/models/test:predict',
                                     headers={"Content-Type": "application/json"}, body=body)
    assert "Invalid JSON" in result.text

@pytest.mark.parametrize(
    "body, expected_format, evaluation", INPUT_FORMAT_CHECK_TEST_CASES
)
def test_get_input_format(mocker, body, expected_format, evaluation):
    evaluate_inputs_mock = mocker.patch(
        'ie_serving.server.rest_msg_validation._evaluate_inputs')
    evaluate_instances_mock = mocker.patch(
        'ie_serving.server.rest_msg_validation._evaluate_instances')

    if evaluation == INPUTS_EVALUATION:
        evaluate_inputs_mock.return_value = expected_format
    elif evaluation == INSTANCES_EVALUATION:
        evaluate_instances_mock.return_value = expected_format

    input_format = get_input_format(body, [DEFAULT_INPUT_KEY])

    if evaluation == INPUTS_EVALUATION:
        evaluate_instances_mock.assert_not_called()
        evaluate_inputs_mock.assert_called_once()
    elif evaluation == INSTANCES_EVALUATION:
        evaluate_instances_mock.assert_called_once()
        evaluate_inputs_mock.assert_not_called()
    else:
        evaluate_instances_mock.assert_not_called()
        evaluate_inputs_mock.assert_not_called()

    assert input_format == expected_format

@pytest.mark.parametrize(
    "inputs, expected_format", EVALUATE_INPUTS_TEST_CASES
)
def test_evaluate_inputs(inputs, expected_format):
    input_format = _evaluate_inputs(inputs)
    assert input_format == expected_format

@pytest.mark.parametrize(
    "instances, expected_format", EVALUATE_INSTANCES_TEST_CASES
)
def test_evaluate_instances(instances, expected_format):
    input_format = _evaluate_instances(instances, [DEFAULT_INPUT_KEY])
    assert input_format == expected_format

