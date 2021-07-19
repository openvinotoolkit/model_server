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

<<<<<<< HEAD
from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusResponse, ModelVersionStatus
from tensorflow.core.protobuf.error_codes_pb2 import Code
from tensorflow_serving.util.status_pb2 import StatusProto
=======
from enum import Enum
>>>>>>> tests + refactor

class CallCount(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3

class ModelState(Enum):
    # UNKNOWN = 0
    START = 10
    LOADING = 20
    AVAILABLE = 30
    UNLOADING = 40
    END = 50

class ErrorCode(Enum):
    OK = 0
    # CANCELLED = 1
    UNKNOWN = 2
    # INVALID_ARGUMENT = 3
    # DEADLINE_EXCEEDED = 4
    # NOT_FOUND = 5
    # ALREADY_EXISTS = 6
    # PERMISSION_DENIED = 7
    # UNAUTHENTICATED = 16
    # RESOURCE_EXHAUSTED = 8
    # FAILED_PRECONDITION = 9
    # ABORTED = 10
    # OUT_OF_RANGE = 11
    # UNIMPLEMENTED = 12
    # INTERNAL = 13
    # UNAVAILABLE = 14
    # DATA_LOSS = 15
    # DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD = 20

MODEL_SPEC_VALID = [
    ("model_name", 1),
    ("3", 17),
    ("*.-", 0),
    ("model_name", 9223372036854775806),
]

MODEL_SPEC_INVALID = [
    (12, 1, TypeError, "model_name type should be string, but is int"),
    (None, -1, TypeError, "model_name type should be string, but is NoneType"),
    (None, 1, TypeError, "model_name type should be string, but is NoneType"),
    ("model_name", "3", TypeError, "model_version type should be int, but is str"),
    ("model_name" , None, TypeError, "model_version type should be int, but is NoneType"),
    ("model_name", -1, ValueError, f'model_version should be in range <0, {2**63-1}>'),
    ("model_name", 9223372036854775809, ValueError, f'model_version should be in range <0, {2**63-1}>'),
]

MODEL_STATUS_RESPONSE_VALID = [
{
    1: {"state" : ModelVersionStatus.State.AVAILABLE, "error_code" : Code.OK, "error_message" : ""}
},
{
    2: {"state" : ModelVersionStatus.State.END, "error_code" : Code.OK, "error_message" : ""},
    3: {"state" : ModelVersionStatus.State.AVAILABLE, "error_code" : Code.OK, "error_message" : ""}
},
{
    1: {"state" : ModelVersionStatus.State.START, "error_code" : Code.OK, "error_message" : ""},
    2: {"state" : ModelVersionStatus.State.LOADING, "error_code" : Code.UNKNOWN, "error_message" : "Could not load CNN"},
    3: {"state" : ModelVersionStatus.State.UNLOADING, "error_code" : Code.OK, "error_message" : ""}
}
]

ADDRESS_VALID = [
    ("localhost"),
    ("19.117.63.126"),
    ("cluster.cloud.iotg.intel.com")
]

ADDRESS_INVALID = [
    (19_117_63_126, TypeError, "address type should be string, but is int"),
    (['localhost'], TypeError, "address type should be string, but is list"),
    ("127.14", ValueError, "address is not valid"),
    ("intel.-com", ValueError, "address is not valid")
]

PORT_VALID = [
    (9000),
    (2**16-1),
]

PORT_INVALID = [
    ("9000", TypeError, "port type should be int, but is type str"),
    ([2**16-1], TypeError, "port type should be int, but is type list"),
    (2**16, ValueError, f"port should be in range <0, {2**16-1}>"),
    (-1, ValueError, f"port should be in range <0, {2**16-1}>")
]

#Special value for testing
PATH_VALID, PATH_INVALID = "valid_path", "invalid_path"

#Special values for testing
SERVER_CERT_PATH_VALID, SERVER_CERT_PATH_INVALID = "server_cert_path_valid", "server_cert_path_invalid"
CLIENT_CERT_PATH_VALID, CLIENT_CERT_PATH_INVALID = "client_cert_path_valid", "client_cert_path_invalid"
CLIENT_KEY_PATH_VALID, CLIENT_KEY_PATH_INVALID = "client_key_path_valid", "client_key_path_invalid"

TLS_CONFIG_VALID = [
    ({"server_cert_path" : PATH_VALID}, CallCount.ONE.value),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : PATH_VALID, "server_cert_path" : PATH_VALID}, CallCount.THREE.value),
]

TLS_CONFIG_INVALID = [
    ({"client_key_path" : PATH_VALID, "server_cert_path" : PATH_VALID,} , ValueError, "none or both client_key_path and client_cert_path are required in tls_config",
    CallCount.ZERO.value, None),
    ({"client_cert_path" : PATH_VALID, "server_cert_path" : PATH_VALID}, ValueError, "none or both client_key_path and client_cert_path are required in tls_config",
    CallCount.ZERO.value, None),
    
    ({"non_client_key_path" : PATH_VALID, "client_cert_path" : PATH_VALID, "server_cert_path" : PATH_VALID}, ValueError, "non_client_key_path is not valid tls_config key",
    CallCount.ZERO.value, [True, True, True]),
    ({"client_key_path" : PATH_VALID, "non_client_cert_path" : PATH_VALID, "server_cert_path" : PATH_VALID}, ValueError, "non_client_cert_path is not valid tls_config key",
    CallCount.ZERO.value, [True, True, True]),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : PATH_VALID, "non_server_cert_path" : PATH_VALID}, ValueError, "server_cert_path is not defined in tls_config",
    CallCount.ZERO.value, [True, True, True]),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : PATH_VALID}, ValueError, "server_cert_path is not defined in tls_config",
    CallCount.ZERO.value, None),

    ({"client_key_path" : 1, "client_cert_path" : PATH_VALID, "server_cert_path" : PATH_VALID}, TypeError, f'client_key_path type should be string but is type int',
    CallCount.ZERO.value, [False, True, True]),
    ({"client_key_path" : [PATH_VALID], "client_cert_path" : PATH_VALID, "server_cert_path" : PATH_VALID}, TypeError, f'client_key_path type should be string but is type list',
    CallCount.ZERO.value, [False, True, True]),
    ({"client_key_path" : 1, "client_cert_path" : PATH_VALID, "server_cert_path" : 1}, TypeError, f'client_key_path type should be string but is type int',
    CallCount.ZERO.value, [False, True, False]),

    ({"client_key_path" : PATH_VALID, "client_cert_path" : 1, "server_cert_path" : PATH_VALID}, TypeError, f'client_cert_path type should be string but is type int',
    CallCount.ONE.value, [True, False, True]),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : 1, "server_cert_path" : 1}, TypeError, f'client_cert_path type should be string but is type int',
    CallCount.ONE.value, [True, False, False]),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : [PATH_VALID], "server_cert_path" : PATH_VALID}, TypeError, f'client_cert_path type should be string but is type list', 
    CallCount.ONE.value, [True, False, True]),

    ({"client_key_path" : PATH_VALID, "client_cert_path" : PATH_VALID, "server_cert_path" : 1}, TypeError, f'server_cert_path type should be string but is type int',
    CallCount.TWO.value, [True, True, False]),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : 1, "server_cert_path" : [PATH_VALID]}, TypeError, f'client_cert_path type should be string but is type int',
    CallCount.ONE.value, [True, False, False]),
    
    ({"client_key_path" : PATH_INVALID, "client_cert_path" : PATH_VALID, "server_cert_path" : PATH_VALID}, ValueError, f'invalid is not valid path to file',
    CallCount.ONE.value, [False, True, True]),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : PATH_INVALID, "server_cert_path" : PATH_VALID}, ValueError, f'/very/invalid is not valid path to file',
    CallCount.TWO.value, [True, False, True]),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : PATH_VALID, "server_cert_path" : PATH_INVALID}, ValueError, f'third_invalid_path is not valid path to file',
    CallCount.THREE.value, [True, True, False]),

]

CONFIG_VALID = [
    ({
        "address" : "localhost",
        "port" : 9000
    }, {"check_address" : CallCount.ONE.value, "check_port" : CallCount.ONE.value, "check_tls_config" : CallCount.ZERO.value}),
    ({
        "address" : "19.117.63.126",
        "port" : 1
    }, {"check_address" : CallCount.ONE.value, "check_port" : CallCount.ONE.value, "check_tls_config" : CallCount.ZERO.value}),
    ({
        "address" : "cluster.cloud.iotg.intel.com",
        "port" : 2**16-1
    }, {"check_address" : CallCount.ONE.value, "check_port" : CallCount.ONE.value, "check_tls_config" : CallCount.ZERO.value}),
    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "server_cert_path" : "valid_path"
        }
    }, {"check_address" : CallCount.ONE.value, "check_port" : CallCount.ONE.value, "check_tls_config" : CallCount.ONE.value}),
    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "client_key_path" : PATH_VALID,
            "client_cert_path" : PATH_VALID,
            "server_cert_path" : PATH_VALID
        }
    }, {"check_address" : CallCount.ONE.value, "check_port" : CallCount.ONE.value, "check_tls_config" : CallCount.ONE.value})
]

CONFIG_INVALID = [
    ({

    },
    {"check_address" : CallCount.ZERO.value, "check_port" : CallCount.ZERO.value, "check_tls_config" : CallCount.ZERO.value},
    ValueError, 'The minimal config must contain address and port', 
    {"check_address" : None, "check_port" : None, "check_tls_config" : None}),
    ({
        "address" : "localhost"
    },
    {"check_address" : CallCount.ZERO.value, "check_port" : CallCount.ZERO.value, "check_tls_config" : CallCount.ZERO.value},
    ValueError, 'The minimal config must contain address and port', 
    {"check_address" : None, "check_port" : None, "check_tls_config" : None}),
    ({
        "port" : 9000
    },
    {"check_address" : CallCount.ZERO.value, "check_port" : CallCount.ZERO.value, "check_tls_config" : CallCount.ZERO.value},
    ValueError, 'The minimal config must contain address and port', 
    {"check_address" : None, "check_port" : None, "check_tls_config" : None}),

    ({
        "address" : ["localhost"],
        "port" : 9000
    },
    {"check_address" : CallCount.ONE.value, "check_port" : CallCount.ZERO.value, "check_tls_config" : CallCount.ZERO.value},
    TypeError, 'address type should be string, but is list', 
    {"check_address" : TypeError('address type should be string, but is list'), "check_port" : None, "check_tls_config" : None}),
    ({
        "address" : "address",
        "port" : '9000'
    },
    {"check_address" : CallCount.ONE.value, "check_port" : CallCount.ZERO.value, "check_tls_config" : CallCount.ZERO.value},
    ValueError, 'address is not valid', 
    {"check_address" : ValueError('address is not valid'), "check_port" : None, "check_tls_config" : None}),
    ({
        "address" : "localhost",
        "port" : '9000'
    },
    {"check_address" : CallCount.ONE.value, "check_port" : CallCount.ONE.value, "check_tls_config" : CallCount.ZERO.value},
    TypeError, 'port type should be int, but is type str', 
    {"check_address" : None, "check_port" : TypeError('port type should be int, but is type str'), "check_tls_config" : None}),
    ({
        "address" : "localhost",
        "port" : 2**16
    },
    {"check_address" : CallCount.ONE.value, "check_port" : CallCount.ONE.value, "check_tls_config" : CallCount.ZERO.value},
    ValueError, f"port should be in range <0, {2**16-1}>", 
    {"check_address" : None, "check_port" : ValueError(f"port should be in range <0, {2**16-1}>"), "check_tls_config" : None}),

    ({
        "address" : "localhost",
        "port" : 2**16,
        "tls_config" : {

        }
    },
    {"check_address" : CallCount.ONE.value, "check_port" : CallCount.ONE.value, "check_tls_config" : CallCount.ONE.value},
    ValueError, 'server_cert_path is not defined in tls_config', 
    {"check_address" : None, "check_port" : None, "check_tls_config" : ValueError('server_cert_path is not defined in tls_config')}),
    ({
        "address" : "localhost",
        "port" : 2**16,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" :PATH_VALID
        }
    },
    {"check_address" : CallCount.ONE.value, "check_port" : CallCount.ONE.value, "check_tls_config" : CallCount.ONE.value},
    ValueError, 'none or both client_key_path and client_cert_path are required in tls_config', 
    {"check_address" : None, "check_port" : None, "check_tls_config" : ValueError('none or both client_key_path and client_cert_path are required in tls_config')}),
    ({
        "address" : "localhost",
        "port" : 2**16,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" :PATH_VALID,
            "client_cert_path" : PATH_VALID,
            "invalid_key_name" : PATH_VALID
        }
    },
    {"check_address" : CallCount.ONE.value, "check_port" : CallCount.ONE.value, "check_tls_config" : CallCount.ONE.value},
    ValueError,  'invalid_key_name is not valid tls_config key', 
    {"check_address" : None, "check_port" : None, "check_tls_config" : ValueError('invalid_key_name is not valid tls_config key')}),
    ({
        "address" : "localhost",
        "port" : 2**16,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" :PATH_VALID,
            "client_cert_path" : 123,
        }
    },
    {"check_address" : CallCount.ONE.value, "check_port" : CallCount.ONE.value, "check_tls_config" : CallCount.ONE.value},
    TypeError,  'client_cert_path type should be string but is type int', 
    {"check_address" : None, "check_port" : None, "check_tls_config" : TypeError('client_cert_path type should be string but is type int')}),
    ({
        "address" : "localhost",
        "port" : 2**16,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" :"invalid_path",
            "client_cert_path" : PATH_VALID,
        }
    },
    {"check_address" : CallCount.ONE.value, "check_port" : CallCount.ONE.value, "check_tls_config" : CallCount.ONE.value},
    ValueError,  'invalid_path is not valid path to file', 
    {"check_address" : None, "check_port" : None, "check_tls_config" : ValueError('invalid_path is not valid path to file')}),
]

CHANNEL_CERTS_VALID = [
    (SERVER_CERT_PATH_VALID, CLIENT_CERT_PATH_VALID, CLIENT_KEY_PATH_VALID,
    {"check_certificate_valid" : CallCount.TWO.value, "check_key_valid" : CallCount.ONE.value}),

    (SERVER_CERT_PATH_VALID, None, None,
    {"check_certificate_valid" : CallCount.ONE.value, "check_key_valid" : CallCount.ZERO.value})
]

CHANNEL_CERTS_INVALID = [
    (SERVER_CERT_PATH_INVALID, None, None,
    {"check_certificate_valid" : CallCount.ONE.value, "check_key_valid" : CallCount.ZERO.value},
    ValueError, f'{SERVER_CERT_PATH_INVALID} file is not valid certificate',
    {"check_certificate_valid" : ValueError(f'{SERVER_CERT_PATH_INVALID} file is not valid certificate'), "check_key_valid" : None}),

    (SERVER_CERT_PATH_INVALID, CLIENT_CERT_PATH_VALID, CLIENT_KEY_PATH_VALID,
    {"check_certificate_valid" : CallCount.ONE.value, "check_key_valid" : CallCount.ZERO.value},
    ValueError, f'{SERVER_CERT_PATH_INVALID} file is not valid certificate',
    {"check_certificate_valid" : ValueError(f'{SERVER_CERT_PATH_INVALID} file is not valid certificate'), "check_key_valid" : None}),

    (SERVER_CERT_PATH_INVALID, CLIENT_CERT_PATH_INVALID, CLIENT_KEY_PATH_VALID,
    {"check_certificate_valid" : CallCount.ONE.value, "check_key_valid" : CallCount.ZERO.value},
    ValueError, f'{SERVER_CERT_PATH_INVALID} file is not valid certificate',
    {"check_certificate_valid" : ValueError(f'{SERVER_CERT_PATH_INVALID} file is not valid certificate'), "check_key_valid" : None}),

    (SERVER_CERT_PATH_VALID, CLIENT_CERT_PATH_INVALID, CLIENT_KEY_PATH_VALID,
    {"check_certificate_valid" : CallCount.TWO.value, "check_key_valid" : CallCount.ZERO.value},
    ValueError, f'{CLIENT_CERT_PATH_INVALID} file is not valid certificate',
    {"check_certificate_valid" : [None, ValueError(f'{CLIENT_CERT_PATH_INVALID} file is not valid certificate')], "check_key_valid" : None}),

    (SERVER_CERT_PATH_INVALID, CLIENT_CERT_PATH_VALID, CLIENT_KEY_PATH_INVALID,
    {"check_certificate_valid" : CallCount.ONE.value, "check_key_valid" : CallCount.ZERO.value},
    ValueError, f'{SERVER_CERT_PATH_INVALID} file is not valid certificate',
    {"check_certificate_valid" : ValueError('client certificate file is not valid'), "check_key_valid" : None}),

    (SERVER_CERT_PATH_VALID, CLIENT_CERT_PATH_VALID, CLIENT_KEY_PATH_INVALID,
    {"check_certificate_valid" : CallCount.TWO.value, "check_key_valid" : CallCount.ONE.value},
    ValueError, f'{CLIENT_KEY_PATH_INVALID} file is not valid private key',
    {"check_certificate_valid" : [None, None], "check_key_valid" : ValueError(f'{CLIENT_KEY_PATH_INVALID} file is not valid private key')}),
]

CERTIFICATE_VALID = [
    (PATH_VALID),
]

CERTIFICATE_INVALID = [
    (PATH_INVALID, ValueError, f'{PATH_INVALID} file is not valid certificate', ValueError(f'{PATH_INVALID} file is not valid certificate')),
]

PRIVATE_KEY_VALID = [
    (PATH_VALID),
]

PRIVATE_KEY_INVALID = [
    (PATH_INVALID, ValueError, f'{PATH_INVALID} file is not valid private key', ValueError(f'{PATH_INVALID} file is not valid private key')),
]

BUILD_VALID = [
    ({
        "address" : "localhost",
        "port" : 9000
    }, {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ZERO.value}),
    ({
        "address" : "19.117.63.126",
        "port" : 1
    }, {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ZERO.value}),
    ({
        "address" : "cluster.cloud.iotg.intel.com",
        "port" : 2**16-1
    }, {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ZERO.value}),
    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "server_cert_path" : "valid_path"
        }
    }, {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ONE.value}),
    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "client_key_path" : PATH_VALID,
            "client_cert_path" : PATH_VALID,
            "server_cert_path" : PATH_VALID
        }
    }, {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ONE.value})
]

BUILD_INVALID = [
    ({

    },
    ValueError, 'The minimal config must contain address and port',
    {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ZERO.value}),
    ({
        "address" : "localhost"
    },
    ValueError, 'The minimal config must contain address and port',
    {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ZERO.value}),
    ({
        "port" : 9000
    },
    ValueError, 'The minimal config must contain address and port',
    {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ZERO.value}),
    ({
        "address" : ["localhost"],
        "port" : 9000
    },
    TypeError, 'address type should be string, but is list',
    {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ZERO.value}),
    ({
        "address" : "address",
        "port" : '9000'
    },
    ValueError, 'address is not valid',
    {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ZERO.value}),
    ({
        "address" : "localhost",
        "port" : '9000'
    },
    TypeError, 'port type should be int, but is type str',
    {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ZERO.value}),
    ({
        "address" : "localhost",
        "port" : 2**16
    },
    ValueError, f"port should be in range <0, {2**16-1}>",
    {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ZERO.value}),

    ({
        "address" : "localhost",
        "port" : 2**16,
        "tls_config" : {

        }
    },
    ValueError, 'server_cert_path is not defined in tls_config',
    {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ZERO.value}),
    ({
        "address" : "localhost",
        "port" : 2**16,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" : PATH_VALID
        }
    },
    ValueError, 'none or both client_key_path and client_cert_path are required in tls_config',
    {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ZERO.value}),
    ({
        "address" : "localhost",
        "port" : 2**16,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" : PATH_VALID,
            "client_cert_path" : PATH_VALID,
            "invalid_key_name" : PATH_VALID
        }
    },
    ValueError,  'invalid_key_name is not valid tls_config key',
    {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ZERO.value}),
    ({
        "address" : "localhost",
        "port" : 2**16,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" :PATH_VALID,
            "client_cert_path" : 123,
        }
    },
    TypeError,  'client_cert_path type should be string but is type int',
    {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ZERO.value}),
    ({
        "address" : "localhost",
        "port" : 2**16,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" : "invalid_path",
            "client_cert_path" : PATH_VALID,
        }
    },
    ValueError,  'invalid_path is not valid path to file',
    {"check_config" : CallCount.ONE.value, "prepare_certs" : CallCount.ZERO.value}),
]
