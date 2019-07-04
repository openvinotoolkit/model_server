from ie_serving.server.constants import INVALID_FORMAT, COLUMN_FORMAT, \
    COLUMN_SIMPLIFIED, ROW_FORMAT, ROW_SIMPLIFIED

JSON_CHECK_TEST_CASES = [
    "just_text", "1234", "[]",
    "{\"key\": \"value\"", "\"key\": \"value\"}", "\"key\": \"value\"",
    "{key: value}", "{1: \"value\"}", "{\"value\"}",
    "{\"key\": 1 \"key2\": \"value\"}",
]

NO_EVALUATION = 0
INPUTS_EVALUATION = 1
INSTANCES_EVALUATION = 2

INPUT_FORMAT_CHECK_TEST_CASES = [
    ({}, INVALID_FORMAT, NO_EVALUATION),
    ({"key": 1, "key2": "value"}, INVALID_FORMAT, NO_EVALUATION),
    ({"inputs": [1, 2, 3], "instances": ["a", "b", "c"]}, INVALID_FORMAT,
     NO_EVALUATION),

    ({"inputs": []}, INVALID_FORMAT, INPUTS_EVALUATION),
    ({"inputs": {}}, INVALID_FORMAT, INPUTS_EVALUATION),
    ({"instances": []}, INVALID_FORMAT, INSTANCES_EVALUATION),
    ({"instances": {}}, INVALID_FORMAT, INSTANCES_EVALUATION),
    ({"instances": {"input": [1, 2, 3]}}, INVALID_FORMAT,
     INSTANCES_EVALUATION),
    ({"inputs": 1}, INVALID_FORMAT, INPUTS_EVALUATION),
    ({"inputs": "value"}, INVALID_FORMAT, INPUTS_EVALUATION),
    ({"instances": 1}, INVALID_FORMAT, INSTANCES_EVALUATION),
    ({"instances": "value"}, INVALID_FORMAT, INSTANCES_EVALUATION),
    ({"instances": [{"wrong_key": [1, 2, 3]}]}, INVALID_FORMAT,
     INSTANCES_EVALUATION),
    ({"instances": [{"input": [1, 2, 3], "wrong_key": [4, 5, 6]}]},
     INVALID_FORMAT, INSTANCES_EVALUATION),
    ({"instances": [{"input": [1, 2, 3]}, {"wrong_key": [4, 5, 6]}]},
     INVALID_FORMAT, INSTANCES_EVALUATION),
    ({"instances": [{"input": [1, 2, 3]},
                    {"input": [1, 2, 3], "wrong_key": [4, 5, 6]}]},
     INVALID_FORMAT, INSTANCES_EVALUATION),
    ({"instances": [{"input": [1, 2, 3]}, {}]}, INVALID_FORMAT,
     INSTANCES_EVALUATION),
    ({"inputs": {"input_key": [1, 2, 3]}}, COLUMN_FORMAT, INPUTS_EVALUATION),
    ({"inputs": {"input_key": [1, 2, 3], "xxx": 1}}, COLUMN_FORMAT,
     INPUTS_EVALUATION),

    ({"inputs": ["abc"]}, COLUMN_SIMPLIFIED, INPUTS_EVALUATION),
    ({"inputs": [1, [2, 3], 4]}, COLUMN_SIMPLIFIED, INPUTS_EVALUATION),

    ({"instances": [{"input": [1, 2, 3]}, {"input": "value"}]},
     ROW_FORMAT, INSTANCES_EVALUATION),
    ({"instances": [{"input": 123}]}, ROW_FORMAT, INSTANCES_EVALUATION),

    ({"key": 1, "instances": ["abc"]}, ROW_SIMPLIFIED, INSTANCES_EVALUATION),
    ({"instances": [1, [2, 3], 4]}, ROW_SIMPLIFIED, INSTANCES_EVALUATION),

]

EVALUATE_INPUTS_TEST_CASES = [
    ([], INVALID_FORMAT),
    ({}, INVALID_FORMAT),
    (1, INVALID_FORMAT),
    ("value", INVALID_FORMAT),

    ({"input_key": [1, 2, 3]}, COLUMN_FORMAT),
    ({"input_key": [1, 2, 3], "xxx": 1}, COLUMN_FORMAT),

    (["abc"], COLUMN_SIMPLIFIED),
    ([1, [2, 3], 4], COLUMN_SIMPLIFIED),
]

EVALUATE_INSTANCES_TEST_CASES = [
    ([], INVALID_FORMAT),
    ({}, INVALID_FORMAT),
    ({"input": [1, 2, 3]}, INVALID_FORMAT),
    (1, INVALID_FORMAT),
    ("value", INVALID_FORMAT),
    ([{"wrong_key": [1, 2, 3]}], INVALID_FORMAT),
    ([{"input": [1, 2, 3], "wrong_key": [4, 5, 6]}], INVALID_FORMAT),
    ([{"input": [1, 2, 3]}, {"wrong_key": [4, 5, 6]}], INVALID_FORMAT),
    ([{"input": [1, 2, 3]}, {"input": [1, 2, 3], "wrong_key": [4, 5, 6]}],
     INVALID_FORMAT),
    ([{"input": [1, 2, 3]}, {}], INVALID_FORMAT),

    ([{"input": [1, 2, 3]}, {"input": "value"}], ROW_FORMAT),
    ([{"input": 123}], ROW_FORMAT),

    (["abc"], ROW_SIMPLIFIED),
    ([1, [2, 3], 4], ROW_SIMPLIFIED),

]

"""
Body as string:

@pytest.mark.parametrize(
    "body",
    [
        "{}",
        "{\"key\": 1, \"key2\": \"value\"}",
        "{\"inputs\": [1,2,3], \"instances\": [\"a\",\"b\",\"c\"]}",
        "{\"inputs\": []}",
        "{\"inputs\": {}}",
        "{\"instances\": []}",
        "{\"instances\": {}}",
        "{\"instances\": {\"input\": [1,2,3]}}",
        "{\"inputs\": 1}",
        "{\"inputs\": \"value\"}",
        "{\"instances\": 1}",
        "{\"instances\": \"value\"}",
        "{\"instances\": [{\"wrong_key\": [1,2,3]}]}",
        "{\"instances\": [{\"input\": [1,2,3], \"wrong_key\": [4,5,6]}]}",
        "{\"instances\": [{\"input\": [1,2,3]},{\"wrong_key\": [4,5,6]}]}",
        "{\"instances\": [{\"input\": [1,2,3]},"
            "{\"input\": [1,2,3]," "\"wrong_key\": [4,5,6]}]}",
        "{\"instances\": [{\"input\": [1,2,3]}, {}]}",

    ]
)

"""
