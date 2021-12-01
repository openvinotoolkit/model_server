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

from enum import IntEnum


class CallCount(IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3


# (model_name, model_version)
MODEL_SPEC_VALID = [
    ("model_name", 1),
    ("3", 17),
    ("*.-", 0),
    ("model_name", 9223372036854775806),
]

# (model_name, model_version, expected_exception, expected_message)
MODEL_SPEC_INVALID = [
    (12, 1, TypeError, "model_name type should be string, but is int"),
    (None, -1, TypeError, "model_name type should be string, but is NoneType"),
    (None, 1, TypeError, "model_name type should be string, but is NoneType"),
    ("model_name", "3", TypeError, "model_version type should be int, but is str"),
    ("model_name", None, TypeError, "model_version type should be int, but is NoneType"),
    ("model_name", -1, ValueError, f'model_version should be in range <0, {2**63-1}>'),
    ("model_name", 9223372036854775809, ValueError,
     f'model_version should be in range <0, {2**63-1}>'),
]


# (address)
ADDRESS_VALID = [
    ("localhost"),
    ("19.117.63.126"),
    ("cluster.cloud.iotg.intel.com")
]

# (address, expected_exception, expected_message)
ADDRESS_INVALID = [
    ("127.14", ValueError, "address is not valid"),
    ("intel.-com", ValueError, "address is not valid"),
    ("192.168.abc.4", ValueError, "address is not valid"),
    ("900.80.70.11", ValueError, "address is not valid"),
    ("....", ValueError, "address is not valid"),
    ("1000.1000.1000.1000", ValueError, "address is not valid"),
    ("19.-117.63.126", ValueError, "address is not valid"),
    ("0.0.0.0.0", ValueError, "address is not valid"),
    ("0.0.0.0.", ValueError, "address is not valid"),
]

# (port)
PORT_VALID = [
    (9000),
    (2**16-1),
]

# (port, expected_exception, expected_message)
PORT_INVALID = [
    ("port", TypeError, "port should be of type int"),
    ([2**16-1], TypeError, "port should be of type int"),
    (2**16, ValueError, f"port should be in range <0, {2**16-1}>"),
    (-1, ValueError, f"port should be in range <0, {2**16-1}>")
]

# Special values for testing
PATH_VALID, PATH_INVALID = "valid_path", "invalid_path"

# Special values for testing
SERVER_CERT_PATH_VALID = "server_cert_path_valid"
SERVER_CERT_PATH_INVALID = "server_cert_path_invalid"
CLIENT_CERT_PATH_VALID = "client_cert_path_valid"
CLIENT_CERT_PATH_INVALID = "client_cert_path_invalid"
CLIENT_KEY_PATH_VALID = "client_key_path_valid"
CLIENT_KEY_PATH_INVALID = "client_key_path_invalid"

# (tls_config_dict, isfile_method_call_count)
TLS_CONFIG_VALID = [
    ({
        "server_cert_path": PATH_VALID
    },
     CallCount.ONE),
    ({
        "client_key_path": PATH_VALID,
        "client_cert_path": PATH_VALID,
        "server_cert_path": PATH_VALID
    },
     CallCount.THREE),
]

# (tls_config_dict,
# expected_exception, expected_message,
# isfile_method_call_count, [is_valid_client_key_path, is_valid_client_cert_path, is_valid_client_server_cert_path])  # noqa: E501
TLS_CONFIG_INVALID = [
    ({
        "client_key_path": PATH_VALID,
        "server_cert_path": PATH_VALID,
    },
     ValueError, "none or both client_key_path and client_cert_path are required in tls_config",
     CallCount.ZERO, None),
    ({
        "client_cert_path": PATH_VALID,
        "server_cert_path": PATH_VALID
    },
     ValueError, "none or both client_key_path and client_cert_path are required in tls_config",
     CallCount.ZERO, None),

    ({
        "non_client_key_path": PATH_VALID,
        "client_cert_path": PATH_VALID,
        "server_cert_path": PATH_VALID
    },
     ValueError, "none or both client_key_path and client_cert_path are required in tls_config",
     CallCount.ZERO, [True, True, True]),
    ({
        "client_key_path": PATH_VALID,
        "non_client_cert_path": PATH_VALID,
        "server_cert_path": PATH_VALID
    },
     ValueError, "none or both client_key_path and client_cert_path are required in tls_config",
     CallCount.ZERO, [True, True, True]),
    ({
        "client_key_path": PATH_VALID,
        "client_cert_path": PATH_VALID,
        "non_server_cert_path": PATH_VALID
    },
     ValueError, "server_cert_path is not defined in tls_config",
     CallCount.ZERO, [True, True, True]),
    ({
        "client_key_path": PATH_VALID,
        "client_cert_path": PATH_VALID
    },
     ValueError, "server_cert_path is not defined in tls_config",
     CallCount.ZERO, None),

    ({
        "client_key_path": 1,
        "client_cert_path": PATH_VALID,
        "server_cert_path": PATH_VALID
    },
     TypeError, 'client_key_path type should be string but is type int',
     CallCount.ZERO, [False, True, True]),
    ({
        "client_key_path": [PATH_VALID],
        "client_cert_path": PATH_VALID,
        "server_cert_path": PATH_VALID
    },
     TypeError, 'client_key_path type should be string but is type list',
     CallCount.ZERO, [False, True, True]),
    ({
        "client_key_path": 1,
        "client_cert_path": PATH_VALID,
        "server_cert_path": 1
    },
     TypeError, 'client_key_path type should be string but is type int',
     CallCount.ZERO, [False, True, False]),

    ({
        "client_key_path": PATH_VALID,
        "client_cert_path": 1,
        "server_cert_path": PATH_VALID
    },
     TypeError, 'client_cert_path type should be string but is type int',
     CallCount.ONE, [True, False, True]),
    ({
        "client_key_path": PATH_VALID,
        "client_cert_path": 1,
        "server_cert_path": 1
    },
     TypeError, 'client_cert_path type should be string but is type int',
     CallCount.ONE, [True, False, False]),
    ({
        "client_key_path": PATH_VALID,
        "client_cert_path": [PATH_VALID],
        "server_cert_path": PATH_VALID
    },
     TypeError, 'client_cert_path type should be string but is type list',
     CallCount.ONE, [True, False, True]),

    ({
        "client_key_path": PATH_VALID,
        "client_cert_path": PATH_VALID,
        "server_cert_path": 1
    },
     TypeError, 'server_cert_path type should be string but is type int',
     CallCount.TWO, [True, True, False]),
    ({
        "client_key_path": PATH_VALID,
        "client_cert_path": 1,
        "server_cert_path": [PATH_VALID]
    },
     TypeError, 'client_cert_path type should be string but is type int',
     CallCount.ONE, [True, False, False]),

    ({
        "client_key_path": PATH_INVALID,
        "client_cert_path": PATH_VALID,
        "server_cert_path": PATH_VALID
    },
     ValueError, 'invalid_path is not valid path to file',
     CallCount.ONE, [False, True, True]),
    ({
        "client_key_path": PATH_VALID,
        "client_cert_path": PATH_INVALID,
        "server_cert_path": PATH_VALID
    },
     ValueError, 'invalid_path is not valid path to file',
     CallCount.TWO, [True, False, True]),
    ({
        "client_key_path": PATH_VALID,
        "client_cert_path": PATH_VALID,
        "server_cert_path": PATH_INVALID
    },
     ValueError, 'invalid_path is not valid path to file',
     CallCount.THREE, [True, True, False]),

]

# (url, method_call_count_dict= {"method_name": CallCount.NumberOfCalls})
URL_VALID = [
    (
        "localhost:9000",
        {
            "_check_address": CallCount.ONE,
            "_check_port": CallCount.ONE
        }
    ),
    (
        "19.117.63.126:1",
        {
            "_check_address": CallCount.ONE,
            "_check_port": CallCount.ONE
        }
    ),
    (
        f"cluster.cloud.iotg.intel.com:{2**16-1}",
        {
            "_check_address": CallCount.ONE,
            "_check_port": CallCount.ONE
        }
    )
]

# (url,
# method_call_count_dict= {"method_name": (CallCount.NumberOfCalls, error_raised)},
# expected_exception, expected_message
URL_INVALID = [
    (
        "localhost",
        {
            "_check_address": (CallCount.ZERO, None),
            "_check_port": (CallCount.ZERO, None)
        },
        ValueError, 'url must be a string in format <address>:<port>'
    ),

    (
        9000,
        {
            "_check_address": (CallCount.ZERO, None),
            "_check_port": (CallCount.ZERO, None)
        },
        TypeError, 'url must be a string in format <address>:<port>',
    ),

    (
        "address:9000",
        {
            "_check_address": (CallCount.ONE, ValueError('address is not valid')),
            "_check_port": (CallCount.ZERO, None)
        },
        ValueError, 'address is not valid'
    ),

    (
        "localhost:string",
        {
            "_check_address": (CallCount.ONE, None),
            "_check_port": (CallCount.ONE, TypeError('port should be of type int'))
        },
        TypeError, 'port should be of type int'
    ),

    (
        "localhost:9000:9001",
        {
            "_check_address": (CallCount.ONE, None),
            "_check_port": (CallCount.ONE, TypeError('port should be of type int'))
        },
        TypeError, 'port should be of type int'
    ),

    (
        "localhost:[9000]",
        {
            "_check_address": (CallCount.ONE, None),
            "_check_port": (CallCount.ONE, TypeError('port should be of type int'))
        },
        TypeError, 'port should be of type int'
    ),

    (
        "localhost:9000abc",
        {
            "_check_address": (CallCount.ONE, None),
            "_check_port": (CallCount.ONE, TypeError('port should be of type int'))
        },
        TypeError, 'port should be of type int'
    ),

    (
        f"localhost:{2**16}",
        {
            "_check_address": (CallCount.ONE, None),
            "_check_port": (CallCount.ONE, ValueError(f"port should be in range <0, {2**16-1}>"))
        },
        ValueError, f"port should be in range <0, {2**16-1}>"
    ),

    (
        "localhost:-1",
        {
            "_check_address": (CallCount.ONE, None),
            "_check_port": (CallCount.ONE, ValueError(f"port should be in range <0, {2**16-1}>"))
        },
        ValueError, f"port should be in range <0, {2**16-1}>"
    )
]

# (server_cert_path, client_cert_path, client_key_path,
# method_call_count_dict= {"method_name": CallCount.NumberOfCalls})
CHANNEL_CERTS_VALID = [
    (SERVER_CERT_PATH_VALID, CLIENT_CERT_PATH_VALID, CLIENT_KEY_PATH_VALID,
     {"check_certificate_valid": CallCount.TWO, "check_key_valid": CallCount.ONE}),

    (SERVER_CERT_PATH_VALID, None, None,
     {"check_certificate_valid": CallCount.ONE, "check_key_valid": CallCount.ZERO})
]

# (certificate_path)
CERTIFICATE_VALID = [
    (PATH_VALID),
]

# (key_path)
PRIVATE_KEY_VALID = [
    (PATH_VALID),
]

MODEL_STATUS_INVALID_PARAMS = [
    # Model name check
    ([("model", "name"), 1, 10], TypeError, "model_name type should be string, but is tuple"),
    # Model version check
    (["model_name", "model_version", 10], TypeError,
        "model_version type should be int, but is str"),
    (["model_name", 2**63, 10], ValueError, f"model_version should be in range <0, {2**63-1}>"),
    (["model_name", -1, 10], ValueError, f"model_version should be in range <0, {2**63-1}>"),
    # Timeout check
    (["model_name", 1, "string"], TypeError, "timeout value must be positive float"),
    (["model_name", 1, 0], TypeError, "timeout value must be positive float"),
    (["model_name", 1, -1], TypeError, "timeout value must be positive float"),
]

MODEL_METADATA_INVALID_PARAMS = MODEL_STATUS_INVALID_PARAMS

# ([inputs, model_name, model_version, timeout], expected_exception, expected_message)
PREDICT_INVALID_PARAMS = [
    # Inputs check
    (
        ["string", "model_name", 1, 1],
        TypeError, "inputs type should be dict, but is str"
    ),
    (
        [1, "model_name", 1, 1],
        TypeError, "inputs type should be dict, but is int"
    ),
    (
        [[1, 2, 3], "model_name", 1, 1],
        TypeError, "inputs type should be dict, but is list"
    ),
    (
        [{1: [1, 2, 3]}, "model_name", 1, 1],
        TypeError, "inputs keys type should be str, but found int"
    ),
    (
        [{"input": [1, 2, "three"]}, "model_name", 1, 1],
        TypeError, "provided values type is not valid"
    ),
    (
        [{"input": [[1, 2], [3, 4, 5]]}, "model_name", 1, 1],
        ValueError, "argument must be a dense tensor: [[1, 2], [3, 4, 5]] "
                    "- got shape [2], but wanted [2, 2]"
    ),
    # Model name check
    ([{"input": 1.0}, ("model", "name"), 1, 10], TypeError,
        "model_name type should be string, but is tuple"),
    # Model version check
    ([{"input": 1.0}, "model_name", "model_version", 10], TypeError,
     "model_version type should be int, but is str"),
    ([{"input": 1.0}, "model_name", 2**63, 10], ValueError,
        f"model_version should be in range <0, {2**63-1}>"),
    ([{"input": 1.0}, "model_name", -1, 10], ValueError,
        f"model_version should be in range <0, {2**63-1}>"),
    # Timeout check
    ([{"input": 1.0}, "model_name", 1, "string"], TypeError,
        "timeout value must be positive float"),
    ([{"input": 1.0}, "model_name", 1, 0], TypeError,
        "timeout value must be positive float"),
    ([{"input": 1.0}, "model_name", 1, -1], TypeError,
        "timeout value must be positive float"),
]
