import pytest

from ie_serving.server.rest_msg_validation import get_input_format, \
    _evaluate_inputs, _evaluate_instances
from config import INPUT_FORMAT_CHECK_TEST_CASES, \
    EVALUATE_INPUTS_TEST_CASES, EVALUATE_INSTANCES_TEST_CASES, \
    INPUTS_EVALUATION, INSTANCES_EVALUATION
# TO DO: Move to another file. Only tests for validation functions here.
from conftest import DEFAULT_INPUT_KEY


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
        assert not evaluate_instances_mock.called
        assert evaluate_inputs_mock.called
    elif evaluation == INSTANCES_EVALUATION:
        assert evaluate_instances_mock.called
        assert not evaluate_inputs_mock.called
    else:
        assert not evaluate_instances_mock.called
        assert not evaluate_inputs_mock.called

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
