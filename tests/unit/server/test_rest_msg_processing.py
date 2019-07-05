import pytest
from config import FORMAT_TRANSLATION_TEST_CASES, \
    PREPROCESS_JSON_TEST_CASES, PREPARE_JSON_RESPONSE_TEST_CASES, \
    DEFAULT_INPUT_KEY

from ie_serving.server.constants import ROW_FORMAT
from ie_serving.server.rest_msg_processing import _row_to_column, \
    _column_to_row, preprocess_json_request, prepare_json_response


@pytest.mark.parametrize(
    "row, column", FORMAT_TRANSLATION_TEST_CASES
)
def test_row_to_column(row, column):
    result = _row_to_column(row)
    assert result == column


@pytest.mark.parametrize(
    "row, column", FORMAT_TRANSLATION_TEST_CASES
)
def test_column_to_row(row, column):
    result = _column_to_row(column)
    assert result == row


@pytest.mark.parametrize(
    "input_format, request_body, expected_output", PREPROCESS_JSON_TEST_CASES
)
def test_preprocess_json_request(mocker, input_format, request_body,
                                 expected_output):
    row_to_column_mock = mocker.patch(
        'ie_serving.server.rest_msg_processing._row_to_column')
    row_to_column_mock.return_value = expected_output
    result = preprocess_json_request(request_body, input_format,
                                     [DEFAULT_INPUT_KEY])
    if input_format == ROW_FORMAT:
        assert row_to_column_mock.called

    assert result == expected_output


@pytest.mark.parametrize(
    "output_representation, inference_output, model_available_outputs, "
    "expected_output", PREPARE_JSON_RESPONSE_TEST_CASES
)
def test_prepare_json_response(output_representation, inference_output,
                               model_available_outputs, expected_output):
    response = prepare_json_response(output_representation, inference_output,
                                     model_available_outputs)
    assert response == expected_output
