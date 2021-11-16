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

# (config_dict,
# method_call_count_dict= {"method_name": CallCount.NumberOfCalls},
# expected_client_key, expected_server_cert)
BUILD_VALID = [
    ({
        "url": "localhost:9000",
    }, {"_check_url": CallCount.ONE, "_check_tls_config": CallCount.ZERO}, None, None),
    ({
        "url": "19.117.63.126:1",
    }, {"_check_url": CallCount.ONE, "_check_tls_config": CallCount.ZERO}, None, None),
    ({
        "url": f"cluster.cloud.iotg.intel.com:{2**16-1}",
    }, {"_check_url": CallCount.ONE, "_check_tls_config": CallCount.ZERO}, None, None),
    ({
        "url": "localhost:9000",
        "tls_config": {
            "server_cert_path": PATH_VALID
        }
    }, {"_check_url": CallCount.ONE, "_check_tls_config": CallCount.ONE}, None, (PATH_VALID,)),
    ({
        "url": "localhost:9000",
        "tls_config": {
            "client_key_path": PATH_VALID,
            "client_cert_path": PATH_VALID,
            "server_cert_path": PATH_VALID
        }
    }, {"_check_url": CallCount.ONE, "_check_tls_config": CallCount.ONE}, (PATH_VALID, PATH_VALID),
        (PATH_VALID, ))
]

# (config_dict,
# method_call_dict= {"method_name": (CallCount.NumberOfCalls, error_raised)},
# expected_exception, expected_message)
BUILD_INVALID_CONFIG = [
    (
        {
            "url": "localhost"
        },
        {
            "_check_url": (CallCount.ONE, ValueError("url must be a string in format "
                                                     "<address>:<port>")),
            "_check_tls_config": (CallCount.ZERO, None),
        },
        ValueError, "url must be a string in format <address>:<port>"
    ),

    (
        {
            "url": 123
        },
        {
            "_check_url": (CallCount.ONE, TypeError("url must be a string in format "
                                                    "<address>:<port>")),
            "_check_tls_config": (CallCount.ZERO, None),
        },
        TypeError, "url must be a string in format <address>:<port>"
    ),

    (
        {
            "url": "address:9000",
        },
        {
            "_check_url": (CallCount.ONE, ValueError("address is not valid")),
            "_check_tls_config": (CallCount.ZERO, None),
        },
        ValueError, "address is not valid"
    ),

    (
        {
            "url": "localhost:port"
        },
        {
            "_check_url": (CallCount.ONE, TypeError("port should be of type int")),
            "_check_tls_config": (CallCount.ZERO, None),
        },
        TypeError, "port should be of type int"
    ),

    (
        {
            "url": f"localhost:{2**16}"
        },
        {
            "_check_url": (CallCount.ONE, ValueError(f"port should be in range <0, {2**16-1}>")),
            "_check_tls_config": (CallCount.ZERO, None),
        },
        ValueError, f"port should be in range <0, {2**16-1}>"
    ),

    (
        {
            "url": "localhost:9000",
            "tls_config": 123
        },
        {
            "_check_url": (CallCount.ONE, None),
            "_check_tls_config": (CallCount.ONE, TypeError("tls_config should be of type dict")),
        },
        TypeError, "tls_config should be of type dict"
    ),

    (
        {
            "url": "localhost:9000",
            "tls_config": {

            }
        },
        {
            "_check_url": (CallCount.ONE, None),
            "_check_tls_config": (CallCount.ONE, ValueError("server_cert_path is not defined "
                                                            "in tls_config")),
        },
        ValueError, "server_cert_path is not defined in tls_config"
    ),

    (
        {
            "url": "10.20.30.40:1000",
            "tls_config": {
                "server_cert_path": PATH_VALID,
                "client_key_path": PATH_VALID
            }
        },
        {
            "_check_url": (CallCount.ONE, None),
            "_check_tls_config": (CallCount.ONE, ValueError("none or both client_key_path and "
                                                            "client_cert_path are required "
                                                            "in tls_config")),
        },
        ValueError, "none or both client_key_path and client_cert_path are required in tls_config"
    ),

    (
        {
            "url": "localhost:9000",
            "tls_config": {
                "server_cert_path": PATH_VALID,
                "client_key_path": PATH_VALID,
                "client_cert_path": PATH_VALID,
                "invalid_key_name": PATH_VALID
            }
        },
        {
            "_check_url": (CallCount.ONE, None),
            "_check_tls_config": (CallCount.ONE, ValueError("invalid_key_name is not valid "
                                                            "tls_config key")),
        },
        ValueError, "invalid_key_name is not valid tls_config key"
    ),

    (
        {
            "url": "localhost:9000",
            "tls_config": {
                "server_cert_path": PATH_VALID,
                "client_key_path": PATH_VALID,
                "client_cert_path": 123,
            }
        },
        {
            "_check_url": (CallCount.ONE, None),
            "_check_tls_config": (CallCount.ONE, TypeError("client_cert_path type should be string "
                                                           "but is type int")),
        },
        TypeError, "client_cert_path type should be string but is type int"
    ),

    (
        {
            "url": "localhost:9000",
            "tls_config": {
                "server_cert_path": PATH_VALID,
                "client_key_path": "invalid_path",
                "client_cert_path": PATH_VALID,
            }
        },
        {
            "_check_url": (CallCount.ONE, None),
            "_check_tls_config": (CallCount.ONE, ValueError("invalid_path is not valid "
                                                            "path to file")),
        },
        ValueError,  "invalid_path is not valid path to file"
    ),

]

MODEL_STATUS_REQUEST_INVALID_REQUEST_TYPE = [
    (requests.Request(), TypeError, 'request type should be HttpModelStatusRequest, '
     'but is Request'),
    ("request", TypeError, 'request type should be HttpModelStatusRequest, '
     'but is str')
]

MODEL_METADATA_REQUEST_INVALID_REQUEST_TYPE = [
    (requests.Request(), TypeError, 'request type should be HttpModelMetadataRequest, '
     'but is Request'),
    ("request", TypeError, 'request type should be HttpModelMetadataRequest, '
     'but is str')
]

PREDICT_REQUEST_INVALID_REQUEST_TYPE = [
    (requests.Request(), TypeError, 'request type should be HttpPredictRequest, '
     'but is Request'),
    ("request", TypeError, 'request type should be HttpPredictRequest, '
     'but is str')
]

# (response_outputs_dict, expected_outputs_dict)
PREDICT_RESPONSE_VALID_OUTPUTS = [
    (
        """
        {
            "outputs": [[1,2,3,4,5]]
        }
        """,
        {
            "outputs": array([[1, 2, 3, 4, 5]])
        }
    ),
    (
        """
        {
            "outputs": {
                "output1": [1,2,3],
                "output2": [4,5,6]
            }
        }
        """,
        {
            "outputs": {
                "output1": array([1, 2, 3]),
                "output2": array([4, 5, 6])
            }
        }
    )
]

# (response_outputs_dict, expected_outputs_dict)
RESPONSE_VALID_OTHER = [
    (
        """
        {
            "error": "Model with requested name is not found"
        }
        """,
        {
            "error": "Model with requested name is not found"
        }
    ),
    (
        """
        {
            "outputs": "string"
        }
        """,
        {
             "outputs": "string"
        }
    ),
    (
        """
        {
            "outputs": 123456789
        }
        """,
        {
             "outputs": 123456789
        }
    ),

    # This should never happen as this is not OVMS API.
    # There's no validation in to_dict().
    # Data not under "outputs" key will be returned as is in dict format.
    # Data under "outputs" key will be additionally converted to numpy
    # to match gRPC to_dict() output.
    (
        """
        {
            "output123": [1,2,3,4,5]
        }
        """,
        {
            "output123": [1, 2, 3, 4, 5]
        }
    ),
    (
        """
        {
            "output123": "string"
        }
        """,
        {
            "output123": "string"
        }
    ),
]


# (response_outputs_dict, expected_outputs_dict)
METADATA_RESPONSE_VALID_OUTPUTS = [
    (
        """
        {
            "modelSpec": {
                "name": "empty",
                "signatureName": "",
                "version": "1"
            },
            "metadata": {
                "signature_def": {
                "@type": "type.googleapis.com/tensorflow.serving.SignatureDefMap",
                "signatureDef": {
                    "serving_default": {
                        "inputs": {
                            "input": {
                                "dtype": "DT_FLOAT",
                                "tensorShape": {
                                    "dim": [
                                        {
                                        "size": "1",
                                        "name": ""
                                        },
                                        {
                                        "size": "10",
                                        "name": ""
                                        }
                                    ],
                                    "unknownRank": false
                                },
                                "name": "input"
                            }
                        },
                        "outputs": {
                            "output": {
                                "dtype": "DT_FLOAT",
                                "tensorShape": {
                                    "dim": [
                                        {
                                            "size": "1",
                                            "name": ""
                                        },
                                        {
                                            "size": "10",
                                            "name": ""
                                        }
                                    ],
                                    "unknownRank": false
                                },
                                "name": "output"
                            }
                        },
                    "methodName": ""
                    }
                }
            }
        }
        }
        """,
        {
            "model_version": 1,
            "inputs": {
                "input": {"dtype": "DT_FLOAT", "shape": [1, 10]}
            },
            "outputs": {
                "output": {"dtype": "DT_FLOAT", "shape": [1, 10]}
            }
        }
    ),

    (
        """
        {
            "modelSpec": {
                "name": "empty",
                "signatureName": "",
                "version": "4"
            },
            "metadata": {
                "signature_def": {
                "@type": "type.googleapis.com/tensorflow.serving.SignatureDefMap",
                "signatureDef": {
                    "serving_default": {
                        "inputs": {
                            "input1": {
                                "dtype": "DT_FLOAT",
                                "tensorShape": {
                                    "dim": [
                                        {
                                        "size": "1",
                                        "name": ""
                                        },
                                        {
                                        "size": "300",
                                        "name": ""
                                        },
                                        {
                                        "size": "300",
                                        "name": ""
                                        }
                                    ],
                                    "unknownRank": false
                                },
                                "name": "input1"
                            },
                            "input2": {
                                "dtype": "DT_FLOAT",
                                "tensorShape": {
                                    "dim": [
                                        {
                                        "size": "1",
                                        "name": ""
                                        },
                                        {
                                        "size": "10",
                                        "name": ""
                                        }
                                    ],
                                    "unknownRank": false
                                },
                                "name": "input2"
                            }
                        },
                        "outputs": {
                            "output1": {
                                "dtype": "DT_INT64",
                                "tensorShape": {
                                    "dim": [
                                        {
                                            "size": "1",
                                            "name": ""
                                        },
                                        {
                                            "size": "1",
                                            "name": ""
                                        }
                                    ],
                                    "unknownRank": false
                                },
                                "name": "output1"
                            },
                            "output2": {
                                "dtype": "DT_FLOAT",
                                "tensorShape": {
                                    "dim": [
                                        {
                                            "size": "1",
                                            "name": ""
                                        },
                                        {
                                            "size": "10",
                                            "name": ""
                                        },
                                        {
                                            "size": "10",
                                            "name": ""
                                        },
                                        {
                                            "size": "10",
                                            "name": ""
                                        }
                                    ],
                                    "unknownRank": false
                                },
                                "name": "output2"
                            }
                        },
                    "methodName": ""
                    }
                }
            }
        }
        }
        """,
        {
            "model_version": 4,
            "inputs": {
                "input1": {"dtype": "DT_FLOAT", "shape": [1, 300, 300]},
                "input2": {"dtype": "DT_FLOAT", "shape": [1, 10]}
            },
            "outputs": {
                "output1": {"dtype": "DT_INT64", "shape": [1, 1]},
                "output2": {"dtype": "DT_FLOAT", "shape": [1, 10, 10, 10]}
            }
        }
    )
]


# (response_outputs_dict, expected_outputs_dict)
STATUS_RESPONSE_VALID_OUTPUTS = [
    (
        """
        {
            "model_version_status": [
                {
                    "version": "1",
                    "state": "AVAILABLE",
                    "status": {
                        "error_code": "OK",
                        "error_message": "OK"
                    }
                }
            ]
        }
        """,
        {
            1: {
                "state": "AVAILABLE",
                "error_code": "OK",
                "error_message": "OK"
            }
        }
    ),
    (
        """
        {
            "model_version_status": [
                {
                    "version": "1",
                    "state": "END",
                    "status": {
                        "error_code": "OK",
                        "error_message": "OK"
                    }
                },
                {
                    "version": "2",
                    "state": "AVAILABLE",
                    "status": {
                        "error_code": "OK",
                        "error_message": "OK"
                    }
                },
                {
                    "version": "3",
                    "state": "LOADING",
                    "status": {
                        "error_code": "UNKNOWN",
                        "error_message": "Could not load model version - topology not supported"
                    }
                }
            ]
        }
        """,
        {
            1: {
                "state": "END",
                "error_code": "OK",
                "error_message": "OK"
            },
            2: {
                "state": "AVAILABLE",
                "error_code": "OK",
                "error_message": "OK"
            },
            3: {
                "state": "LOADING",
                "error_code": "UNKNOWN",
                "error_message": "Could not load model version - topology not supported"
            }
        }
    ),
]
