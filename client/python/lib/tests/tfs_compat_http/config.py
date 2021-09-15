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

import json
import requests
from numpy import array, int32, float32, float128

from config import CallCount, PATH_VALID # noqa

# (inputs_dict,
# model_name, model_version, expected_exception, expected_message)
PREDICT_REQUEST_INVALID_INPUTS = [
    ([],
     'model_name', 0, TypeError, "inputs type should be dict, but is list"),
    (('input1', [1, 2, 3]),
     'model_name', 0, TypeError, "inputs type should be dict, but is tuple"),
    ({
        1: [1, 2, 3],
        "input2": [1, 2]
    }, 'model_name', 0, TypeError, "inputs keys should be type str, but found int"),
    ({
        "input1": [[1.0, 2.0], [1.0, 2.0, 3.0]]
    }, 'model_name', 0, ValueError,
     ("argument must be a dense tensor: [[1.0, 2.0], [1.0, 2.0, 3.0]] - "
      "got shape [2], but wanted [2, 2]")),
    ({
        "input1": [[(1, 2, 3)], [(1, 2)], [(1, 2, 3)]]
    }, 'model_name', 0, TypeError, "provided values type is not valid"),
    ({
        "input1": float128(2.5)
    }, 'model_name', 0, TypeError, "provided values type is not valid"),
    ({
        "input1": (1, 2, 3)
    }, 'model_name', 0, TypeError,
     "values type should be (list, np.ndarray, scalar), but is tuple"),
    ({
        "input1": [
            [bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00]),
             bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00])],
            [bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00]),
             bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00])]
        ]
    }, 'model_name', 0, ValueError, "bytes values with dtype DT_STRING must be in shape [N]"),
]

# (inputs_dict,
# expected_parsed_input,
# model_name, model_version)
PREDICT_REQUEST_VALID = [
    ({
        "input1": [1, 2, 3],
        "input2": array([1.0, 2.0, 3.0]),
        "input3": [[int32(3), int32(1)], [int32(4), int32(16)]],
    }, json.dumps({
        "inputs": {
            "input1": [1, 2, 3],
            "input2": [1.0, 2.0, 3.0],
            "input3": [[3, 1], [4, 16]]
        }
    }), 'model_name', 0),

    ({
        "input1": 5.0,
        "input2": bytes([1, 2, 3])
    }, json.dumps({
        "inputs": {
            "input1": [5.0],
            "input2": {"b64": ["AQID"]}
        }
    }), 'model_name', 0),

    ({

    }, json.dumps({
        "inputs": {}
    }), 'model_name', 0)
]

# (inputs_dict,
# expected_parsed_input)
PARSE_INPUT_DATA_VALID = [
    (array([[[1, 2, 3]]], dtype=float32), [[[1.0, 2.0, 3.0]]]),

    (array([-1.0, -2.0, -3.0]), [-1.0, -2.0, -3.0]),

    ([[int32(3), int32(1)], [int32(4), int32(16)]], [[3, 1], [4, 16]]),

    (array([[0.012, -0.0002, 0.31, 0.0000014, -0.00054]]),
     [[0.012, -0.0002, 0.31, 0.0000014, -0.00054]]),

    (1, [1]),

    (0, [0]),

    ([1, 2, 3.0], [1.0, 2.0, 3.0]),

    ([bytes([1, 2, 3]), bytes([4, 5, 6]), bytes([7, 8, 9])], {"b64": ["AQID", "BAUG", "BwgJ"]})
]

# (inputs_dict,
# expected_exception, expected_message)
PARSE_INPUT_DATA_INVALID = [
    ([[1.0, 2.0], [1.0, 2.0, 3.0]], ValueError,
     ("argument must be a dense tensor: [[1.0, 2.0], [1.0, 2.0, 3.0]] - "
      "got shape [2], but wanted [2, 2]")),

    ([[(1, 2, 3)], [(1, 2)], [(1, 2, 3)]],
     TypeError, "provided values type is not valid"),

    ([1, 2, 3, "str"],
     TypeError, "provided values type is not valid"),

    ([[1, 2], [3, 4], ["five", 6]],
     TypeError, "provided values type is not valid"),

    (float128(2.5), TypeError, "provided values type is not valid"),

    ((1, 2, 3), TypeError,
     "values type should be (list, np.ndarray, scalar), but is tuple"),

    ([
            [bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00]),
             bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00])],
            [bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00]),
             bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00])]
     ], ValueError, "bytes values with dtype DT_STRING must be in shape [N]"),
]

MODEL_STATUS_REQUEST_INVALID_REQUEST_TYPE = [
    (requests.Request(), TypeError, 'request type should be HttpModelStatusRequest, '
     'but is Request'),
    ("request", TypeError, 'request type should be HttpModelStatusRequest, '
     'but is str')
]
