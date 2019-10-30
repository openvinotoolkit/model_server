import numpy as np

from ie_serving.models.shape_management.utils import ShapeMode
from ie_serving.server.constants import INVALID_FORMAT, COLUMN_FORMAT, \
    COLUMN_SIMPLIFIED, ROW_FORMAT, ROW_SIMPLIFIED

DEFAULT_INPUT_KEY = 'input'
DEFAULT_OUTPUT_KEY = 'output'

JSON_CHECK_TEST_CASES = [
    "just_text", "1234", "[]",
    "{\"key\": \"value\"", "\"key\": \"value\"}", "\"key\": \"value\"",
    "{key: value}", "{1: \"value\"}", "{\"value\"}",
    "{\"key\": 1 \"key2\": \"value\"}"
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

FORMAT_TRANSLATION_TEST_CASES = [
    (
        [
            {
                "in1": [1, 2, 3],
                "in2": ["a", "b", "c"],
                "in3": [[4, 5], [6, 7]]
            },
            {
                "in1": [8, 9, 10],
                "in2": ["d", "e", "f"],
                "in3": [[11, 12], [13, 14]]
            },
            {
                "in1": [15, 16, 17],
                "in2": ["g", "h", "i"],
                "in3": [[18, 19], [20, 21]]
            },
        ],
        {
            "in1": [[1, 2, 3], [8, 9, 10], [15, 16, 17]],
            "in2": [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]],
            "in3": [[[4, 5], [6, 7]], [[11, 12], [13, 14]],
                    [[18, 19], [20, 21]]]
        }
    ),
    (
        [
            {
                "in1": [1, 2, 3],
                "in2": ["a", "b", "c"],
                "in3": [[4, 5], [6, 7]]
            }
        ],
        {
            "in1": [[1, 2, 3]],
            "in2": [["a", "b", "c"]],
            "in3": [[[4, 5], [6, 7]]]
        }
    ),
    (
        [
            {
                "in": [1, 2, 3]
            }
        ],
        {
            "in": [[1, 2, 3]]
        }
    ),
]

PREPROCESS_JSON_TEST_CASES = [
    (ROW_FORMAT, {"instances": FORMAT_TRANSLATION_TEST_CASES[0][0]},
     FORMAT_TRANSLATION_TEST_CASES[0][1]),
    (ROW_SIMPLIFIED, {"instances": [[1, 2, 3, 4]]},
     {DEFAULT_INPUT_KEY: [[1, 2, 3, 4]]}),
    (COLUMN_FORMAT, {"inputs": FORMAT_TRANSLATION_TEST_CASES[1][1]},
     FORMAT_TRANSLATION_TEST_CASES[1][1]),
    (COLUMN_SIMPLIFIED, {"inputs": [[1, 2, 3, 4]]},
     {DEFAULT_INPUT_KEY: [[1, 2, 3, 4]]})
]

PREPARE_JSON_RESPONSE_TEST_CASES = [
    ("row", {"output": [1, 2, 3, 4]}, {"output": "output"},
     {"predictions": [1, 2, 3, 4]}),
    ("row", {"out1": [[1, 2]], "out2": [[3, 4]]},
     {"output1": "out1", "output2": "out2"},
     {"predictions": [{"output1": [1, 2], "output2": [3, 4]}]}),
    ("column", {"output": [1, 2, 3, 4]}, {"output": "output"},
     {"outputs": [1, 2, 3, 4]}),
    ("column", {"out1": [1, 2], "out2": [3, 4]},
     {"output1": "out1", "output2": "out2"},
     {"outputs": {"output1": [1, 2], "output2": [3, 4]}}),
]

DETECT_SHAPES_INCOMPATIBILITY_TEST_CASES = [
    (ShapeMode.AUTO, {'input': (1, 1, 1)},
     {'input': (1, 1, 1)}),
    (ShapeMode.AUTO, {}, None),
    (ShapeMode.DISABLED, {'input': (1, 1, 1)}, 1)
]

SCAN_INPUT_SHAPES_TEST_CASES = [
    ({"input": (1, 1, 1)}, {"input": np.zeros((1, 2, 2))},
     {"input": (1, 2, 2)}),
    ({"input": (1, 1, 1)}, {"input": np.zeros((1, 1, 1))},
     {}),
    ({"input1": (1, 1, 1), "input2": (1, 1, 1)},
     {"input1": np.zeros((1, 2, 2)), "input2": np.zeros((2, 1, 1))},
     {"input1": (1, 2, 2), "input2": (2, 1, 1)}),
    ({"input1": (1, 1, 1), "input2": (1, 1, 1)},
     {"input1": np.zeros((1, 1, 1)), "input2": np.zeros((2, 1, 1))},
     {"input2": (2, 1, 1)}),
    ({"input1": (1, 1, 1), "input2": (1, 1, 1)},
     {"input1": np.zeros((1, 1, 1)), "input2": np.zeros((1, 1, 1))},
     {}),
]

NOT_CALLED = -1

RESHAPE_TEST_CASES = [
    ({"input": (1, 1, 1)},
     {'_reshape': True, '_change_batch_size': False},
     {'_reshape': None, '_change_batch_size': NOT_CALLED},
     None),
    ({"input": (1, 1, 1)},
     {'_reshape': True, '_change_batch_size': False},
     {'_reshape': "Error", '_change_batch_size': NOT_CALLED},
     "Error"),
    (1,
     {'_reshape': False, '_change_batch_size': True},
     {'_reshape': NOT_CALLED, '_change_batch_size': None},
     None),
    (1,
     {'_reshape': False, '_change_batch_size': True},
     {'_reshape': NOT_CALLED, '_change_batch_size': "Error"},
     "Error"),
    ("string",
     {'_reshape': False, '_change_batch_size': False},
     {'_reshape': NOT_CALLED, '_change_batch_size': NOT_CALLED},
     "Unknown error occurred in input reshape preparation"),
    ((1, 1, 1),
     {'_reshape': False, '_change_batch_size': False},
     {'_reshape': NOT_CALLED, '_change_batch_size': NOT_CALLED},
     "Unknown error occurred in input reshape preparation"),
]

PROCESS_SHAPE_PARAM_TEST_CASES = [
    ({"input": "[1, 1, 1]"},
     {"input": (1, 1, 1)},
     {'get_shape_dict': True,
      'get_shape_from_string': False,
      '_shape_as_dict': False},
     {'get_shape_dict': {"input": (1, 1, 1)},
      'get_shape_from_string': NOT_CALLED,
      '_shape_as_dict': NOT_CALLED},
     (ShapeMode.FIXED, {"input": (1, 1, 1)})
     ),
    ({"input": "[1, 1, 1]"},
     {"input": (1, 1, 1)},
     {'get_shape_dict': True,
      'get_shape_from_string': False,
      '_shape_as_dict': False},
     {'get_shape_dict': None,
      'get_shape_from_string': NOT_CALLED,
      '_shape_as_dict': NOT_CALLED},
     (ShapeMode.DEFAULT, None)
     ),
    ("auto",
     {"input": (1, 1, 1)},
     {'get_shape_dict': False,
      'get_shape_from_string': True,
      '_shape_as_dict': False},
     {'get_shape_dict': NOT_CALLED,
      'get_shape_from_string': (ShapeMode.AUTO, None),
      '_shape_as_dict': NOT_CALLED},
     (ShapeMode.AUTO, None)
     ),
    ("bad_param",
     {"input": (1, 1, 1)},
     {'get_shape_dict': False,
      'get_shape_from_string': True,
      '_shape_as_dict': False},
     {'get_shape_dict': NOT_CALLED,
      'get_shape_from_string': (ShapeMode.DEFAULT, None),
      '_shape_as_dict': NOT_CALLED},
     (ShapeMode.DEFAULT, None)
     ),
    ("{\"input\": \"[1, 1, 1]\"}",
     {"input": (1, 1, 1)},
     {'get_shape_dict': False,
      'get_shape_from_string': True,
      '_shape_as_dict': False},
     {'get_shape_dict': NOT_CALLED,
      'get_shape_from_string': (ShapeMode.FIXED, {"input": (1, 1, 1)}),
      '_shape_as_dict': NOT_CALLED},
     (ShapeMode.FIXED, {"input": (1, 1, 1)})
     ),
    ("(1, 1, 1)",
     {"input": (1, 1, 1)},
     {'get_shape_dict': False,
      'get_shape_from_string': True,
      '_shape_as_dict': True},
     {'get_shape_dict': NOT_CALLED,
      'get_shape_from_string': (ShapeMode.FIXED, (1, 1, 1)),
      '_shape_as_dict': {"input": (1, 1, 1)}},
     (ShapeMode.FIXED, {"input": (1, 1, 1)})
     )
]

PROCESS_GET_SHAPE_FROM_STRING_TEST_CASES = [
    ("auto",
     {'load_shape': False,
      'get_shape_tuple': False,
      'get_shape_dict': False},
     {'load_shape': NOT_CALLED,
      'get_shape_tuple': NOT_CALLED,
      'get_shape_dict': NOT_CALLED},
     (ShapeMode.AUTO, None)
     ),
    ("bad_param",
     {'load_shape': True,
      'get_shape_tuple': False,
      'get_shape_dict': False},
     {'load_shape': None,
      'get_shape_tuple': NOT_CALLED,
      'get_shape_dict': NOT_CALLED},
     (ShapeMode.DEFAULT, None)
     ),
    ("(1, 1, 1)",
     {'load_shape': True,
      'get_shape_tuple': True,
      'get_shape_dict': False},
     {'load_shape': [1, 1, 1],
      'get_shape_tuple': (1, 1, 1),
      'get_shape_dict': NOT_CALLED},
     (ShapeMode.FIXED, (1, 1, 1))
     ),
    ("(1, 1, 1)",
     {'load_shape': True,
      'get_shape_tuple': True,
      'get_shape_dict': False},
     {'load_shape': ["string", 1, 1],
      'get_shape_tuple': None,
      'get_shape_dict': NOT_CALLED},
     (ShapeMode.DEFAULT, None)
     ),
    ("{\"input\": \"(1, 1, 1)\"}",
     {'load_shape': True,
      'get_shape_tuple': False,
      'get_shape_dict': True},
     {'load_shape': {"input": [1, 1, 1]},
      'get_shape_tuple': NOT_CALLED,
      'get_shape_dict': {"input": (1, 1, 1)}},
     (ShapeMode.FIXED, {"input": (1, 1, 1)})
     ),
    ("{\"input\": \"(1, 1, 1)\"}",
     {'load_shape': True,
      'get_shape_tuple': False,
      'get_shape_dict': True},
     {'load_shape': {"input": ["string", 1, 1]},
      'get_shape_tuple': NOT_CALLED,
      'get_shape_dict': None},
     (ShapeMode.DEFAULT, None)
     ),
]

PARSE_CONFIG_TEST_CASES = [
    (False, {
        "model_config_list": [
            {
                "config": {
                    "name": "resnet_V1_50",
                    "base_path": "/opt/ml/resnet_V1_50",
                }
            },
            {
                "config": {
                    "name": "pnasnet_large",
                    "base_path": "/opt/ml/pnasnet_large",
                    "batch_size": "auto",
                    "shape": "auto",
                    "target_device": "HDDL",
                    "nireq": 4,
                    "model_version_policy":
                        {"specific": {"versions": [1, 2, 3]}}
                }
            },
            {
                "config": {
                    "name": "pnasnet_large",
                    "base_path": "/opt/ml/pnasnet_large",
                    "batch_size": 4,
                    "shape": {"input": "(1, 2, 3, 4)"},
                    "plugin_config": {"key": "value"}
                }
            }]
    }),
    (True, {
        "model_config_list": [
            {
                "config": {
                    "name": "resnet_V1_50",
                    "base_path": "/opt/ml/resnet_V1_50",
                }
            },
            {
                "config": {
                    "name": "pnasnet_large"
                }
            }
        ]
    }),
    (True, {
        "model_config_list": [
            {
                "config": {
                    "name": "resnet_V1_50",
                    "base_path": "/opt/ml/resnet_V1_50",
                    "plugin_config": 1
                }
            }
        ]
    }),
    (True, {
        "model_config_list": [
            {
                "config": {
                    "name": "resnet_V1_50",
                    "base_path": "/opt/ml/resnet_V1_50",
                    "model_version_policy": "{specific: {versions: [1,2]}}"
                }
            }
        ]
    }),
    (True, {
        "model_config_list": [
            {
                "config": {
                    "name": "resnet_V1_50",
                    "base_path": "/opt/ml/resnet_V1_50",
                    "nireq": "32"
                }
            }
        ]
    }),
    (True, {
        "model_config_list": [
            {
                "config": {
                    "name": "resnet_V1_50",
                    "base_path": "/opt/ml/resnet_V1_50",
                    "target_device": 1
                }
            }
        ]
    }),
    (True, {
        "model_config_list": [
            {
                "config": {
                    "name": "resnet_V1_50",
                    "base_path": "/opt/ml/resnet_V1_50",
                    "shape": 1
                }
            }
        ]
    })
]
