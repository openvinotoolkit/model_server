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

from numpy import array, float64, int32, int8
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from tensorflow_serving.apis.get_model_metadata_pb2 import GetModelMetadataRequest
from tensorflow_serving.apis.get_model_status_pb2 import ModelVersionStatus
from tensorflow.core.framework.types_pb2 import DataType
from tensorflow.core.protobuf.error_codes_pb2 import Code
from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusRequest
from enum import IntEnum
from numpy import array, float128, int32

from ovmsclient.tfs_compat.grpc.tensors import make_tensor_proto
from grpc import StatusCode

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
    ("model_name" , None, TypeError, "model_version type should be int, but is NoneType"),
    ("model_name", -1, ValueError, f'model_version should be in range <0, {2**63-1}>'),
    ("model_name", 9223372036854775809, ValueError, f'model_version should be in range <0, {2**63-1}>'),
]

# responses_dict = {
#    model_version : { expected_status }
# }
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

# (address)
ADDRESS_VALID = [
    ("localhost"),
    ("19.117.63.126"),
    ("cluster.cloud.iotg.intel.com")
]

# (address, expected_exception, expected_message)
ADDRESS_INVALID = [
    (19_117_63_126, TypeError, "address type should be string, but is int"),
    (['localhost'], TypeError, "address type should be string, but is list"),
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
    ("9000", TypeError, "port type should be int, but is type str"),
    ([2**16-1], TypeError, "port type should be int, but is type list"),
    (2**16, ValueError, f"port should be in range <0, {2**16-1}>"),
    (-1, ValueError, f"port should be in range <0, {2**16-1}>")
]

# Special values for testing
PATH_VALID, PATH_INVALID = "valid_path", "invalid_path"

# Special values for testing
SERVER_CERT_PATH_VALID, SERVER_CERT_PATH_INVALID = "server_cert_path_valid", "server_cert_path_invalid"
CLIENT_CERT_PATH_VALID, CLIENT_CERT_PATH_INVALID = "client_cert_path_valid", "client_cert_path_invalid"
CLIENT_KEY_PATH_VALID, CLIENT_KEY_PATH_INVALID = "client_key_path_valid", "client_key_path_invalid"

# (tls_config_dict, isfile_method_call_count)
TLS_CONFIG_VALID = [
    ({
        "server_cert_path" : PATH_VALID
    }, 
    CallCount.ONE),
    ({
        "client_key_path" : PATH_VALID, 
        "client_cert_path" : PATH_VALID, 
        "server_cert_path" : PATH_VALID
    }, 
    CallCount.THREE),
]

# (tls_config_dict,
# expected_exception, expected_message,
# isfile_method_call_count, [is_valid_client_key_path, is_valid_client_cert_path, is_valid_client_server_cert_path])
TLS_CONFIG_INVALID = [
    ({
        "client_key_path" : PATH_VALID, 
        "server_cert_path" : PATH_VALID,
    }, 
    ValueError, "none or both client_key_path and client_cert_path are required in tls_config",
    CallCount.ZERO, None),
    ({
        "client_cert_path" : PATH_VALID, 
        "server_cert_path" : PATH_VALID
    }, 
    ValueError, "none or both client_key_path and client_cert_path are required in tls_config",
    CallCount.ZERO, None),
    
    ({
        "non_client_key_path" : PATH_VALID, 
        "client_cert_path" : PATH_VALID, 
        "server_cert_path" : PATH_VALID
    }, 
    ValueError, "none or both client_key_path and client_cert_path are required in tls_config",
    CallCount.ZERO, [True, True, True]),
    ({
        "client_key_path" : PATH_VALID, 
        "non_client_cert_path" : PATH_VALID, 
        "server_cert_path" : PATH_VALID
    }, 
    ValueError, "none or both client_key_path and client_cert_path are required in tls_config",
    CallCount.ZERO, [True, True, True]),
    ({
        "client_key_path" : PATH_VALID, 
        "client_cert_path" : PATH_VALID, 
        "non_server_cert_path" : PATH_VALID
    }, 
    ValueError, "server_cert_path is not defined in tls_config",
    CallCount.ZERO, [True, True, True]),
    ({
        "client_key_path" : PATH_VALID, 
        "client_cert_path" : PATH_VALID
    }, 
    ValueError, "server_cert_path is not defined in tls_config",
    CallCount.ZERO, None),

    ({
        "client_key_path" : 1, 
        "client_cert_path" : PATH_VALID, 
        "server_cert_path" : PATH_VALID
    }, 
    TypeError, f'client_key_path type should be string but is type int',
    CallCount.ZERO, [False, True, True]),
    ({
        "client_key_path" : [PATH_VALID], 
        "client_cert_path" : PATH_VALID, 
        "server_cert_path" : PATH_VALID
    }, 
    TypeError, f'client_key_path type should be string but is type list',
    CallCount.ZERO, [False, True, True]),
    ({
        "client_key_path" : 1, 
        "client_cert_path" : PATH_VALID, 
        "server_cert_path" : 1
    }, 
    TypeError, f'client_key_path type should be string but is type int',
    CallCount.ZERO, [False, True, False]),

    ({
        "client_key_path" : PATH_VALID, 
        "client_cert_path" : 1, 
        "server_cert_path" : PATH_VALID
    }, 
    TypeError, f'client_cert_path type should be string but is type int',
    CallCount.ONE, [True, False, True]),
    ({
        "client_key_path" : PATH_VALID, 
        "client_cert_path" : 1, 
        "server_cert_path" : 1
    }, 
    TypeError, f'client_cert_path type should be string but is type int',
    CallCount.ONE, [True, False, False]),
    ({
        "client_key_path" : PATH_VALID, 
        "client_cert_path" : [PATH_VALID], 
        "server_cert_path" : PATH_VALID
    }, 
    TypeError, f'client_cert_path type should be string but is type list', 
    CallCount.ONE, [True, False, True]),

    ({
        "client_key_path" : PATH_VALID, 
        "client_cert_path" : PATH_VALID, 
        "server_cert_path" : 1
    }, 
    TypeError, f'server_cert_path type should be string but is type int',
    CallCount.TWO, [True, True, False]),
    ({
        "client_key_path" : PATH_VALID, 
        "client_cert_path" : 1, 
        "server_cert_path" : [PATH_VALID]
    }, 
    TypeError, f'client_cert_path type should be string but is type int',
    CallCount.ONE, [True, False, False]),
    
    ({
        "client_key_path" : PATH_INVALID, 
        "client_cert_path" : PATH_VALID, 
        "server_cert_path" : PATH_VALID
    }, 
    ValueError, f'invalid_path is not valid path to file',
    CallCount.ONE, [False, True, True]),
    ({
        "client_key_path" : PATH_VALID, 
        "client_cert_path" : PATH_INVALID, 
        "server_cert_path" : PATH_VALID
    }, 
    ValueError, f'invalid_path is not valid path to file',
    CallCount.TWO, [True, False, True]),
    ({
        "client_key_path" : PATH_VALID, 
        "client_cert_path" : PATH_VALID, 
        "server_cert_path" : PATH_INVALID
    }, 
    ValueError, f'invalid_path is not valid path to file',
    CallCount.THREE, [True, True, False]),

]

# (config_dict, method_call_count_dict= {"method_name" : CallCount.NumberOfCalls})
CONFIG_VALID = [
    ({
        "address" : "localhost",
        "port" : 9000
    }, {"check_address" : CallCount.ONE, "check_port" : CallCount.ONE, "check_tls_config" : CallCount.ZERO}),
    ({
        "address" : "19.117.63.126",
        "port" : 1
    }, {"check_address" : CallCount.ONE, "check_port" : CallCount.ONE, "check_tls_config" : CallCount.ZERO}),
    ({
        "address" : "cluster.cloud.iotg.intel.com",
        "port" : 2**16-1
    }, {"check_address" : CallCount.ONE, "check_port" : CallCount.ONE, "check_tls_config" : CallCount.ZERO}),
    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "server_cert_path" : "valid_path"
        }
    }, {"check_address" : CallCount.ONE, "check_port" : CallCount.ONE, "check_tls_config" : CallCount.ONE}),
    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "client_key_path" : PATH_VALID,
            "client_cert_path" : PATH_VALID,
            "server_cert_path" : PATH_VALID
        }
    }, {"check_address" : CallCount.ONE, "check_port" : CallCount.ONE, "check_tls_config" : CallCount.ONE})
]

# (config_dict, method_call_count_dict= {"method_name" : CallCount.NumberOfCalls},
# expected_exception, expected_message,
# side_effect_dict = {"method_name" : method_name_side_effect})
CONFIG_INVALID = [
    ({

    },
    {"check_address" : CallCount.ZERO, "check_port" : CallCount.ZERO, "check_tls_config" : CallCount.ZERO},
    ValueError, 'The minimal config must contain address and port', 
    {
        "check_address" : None, 
        "check_port" : None, 
        "check_tls_config" : None
    }),

    ({
        "address" : "localhost"
    },
    {"check_address" : CallCount.ZERO, "check_port" : CallCount.ZERO, "check_tls_config" : CallCount.ZERO},
    ValueError, 'The minimal config must contain address and port', 
    {
        "check_address" : None, 
        "check_port" : None, 
        "check_tls_config" : None
    }),

    ({
        "port" : 9000
    },
    {"check_address" : CallCount.ZERO, "check_port" : CallCount.ZERO, "check_tls_config" : CallCount.ZERO},
    ValueError, 'The minimal config must contain address and port', 
    {
        "check_address" : None, 
        "check_port" : None, 
        "check_tls_config" : None
    }),

    ({
        "address" : ["localhost"],
        "port" : 9000
    },
    {"check_address" : CallCount.ONE, "check_port" : CallCount.ZERO, "check_tls_config" : CallCount.ZERO},
    TypeError, 'address type should be string, but is list', 
    {
        "check_address" : TypeError('address type should be string, but is list'), 
        "check_port" : None, 
        "check_tls_config" : None
    }),
    
    ({
        "address" : "address",
        "port" : '9000'
    },
    {"check_address" : CallCount.ONE, "check_port" : CallCount.ZERO, "check_tls_config" : CallCount.ZERO},
    ValueError, 'address is not valid', 
    {
        "check_address" : ValueError('address is not valid'), 
        "check_port" : None, 
        "check_tls_config" : None
    }),

    ({
        "address" : "localhost",
        "port" : '9000'
    },
    {"check_address" : CallCount.ONE, "check_port" : CallCount.ONE, "check_tls_config" : CallCount.ZERO},
    TypeError, 'port type should be int, but is type str', 
    {
        "check_address" : None, 
        "check_port" : TypeError('port type should be int, but is type str'), 
        "check_tls_config" : None
    }),

    ({
        "address" : "localhost",
        "port" : 2**16
    },
    {"check_address" : CallCount.ONE, "check_port" : CallCount.ONE, "check_tls_config" : CallCount.ZERO},
    ValueError, f"port should be in range <0, {2**16-1}>", 
    {
        "check_address" : None, 
        "check_port" : ValueError(f"port should be in range <0, {2**16-1}>"), 
        "check_tls_config" : None
    }),

    ({
        "address" : "localhost",
        "port" : 2**16,
        "tls_config" : {

        }
    },
    {"check_address" : CallCount.ONE, "check_port" : CallCount.ONE, "check_tls_config" : CallCount.ONE},
    ValueError, 'server_cert_path is not defined in tls_config', 
    {
        "check_address" : None, 
        "check_port" : None, 
        "check_tls_config" : ValueError('server_cert_path is not defined in tls_config')
    }),

    ({
        "address" : "localhost",
        "port" : 2**16,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" :PATH_VALID
        }
    },
    {"check_address" : CallCount.ONE, "check_port" : CallCount.ONE, "check_tls_config" : CallCount.ONE},
    ValueError, 'none or both client_key_path and client_cert_path are required in tls_config', 
    {
        "check_address" : None, 
        "check_port" : None, 
        "check_tls_config" : ValueError('none or both client_key_path and client_cert_path are required in tls_config')
    }),

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
    {"check_address" : CallCount.ONE, "check_port" : CallCount.ONE, "check_tls_config" : CallCount.ONE},
    ValueError,  'invalid_key_name is not valid tls_config key', 
    {
        "check_address" : None, 
        "check_port" : None, 
        "check_tls_config" : ValueError('invalid_key_name is not valid tls_config key')
    }),
    
    ({
        "address" : "localhost",
        "port" : 2**16,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" :PATH_VALID,
            "client_cert_path" : 123,
        }
    },
    {"check_address" : CallCount.ONE, "check_port" : CallCount.ONE, "check_tls_config" : CallCount.ONE},
    TypeError,  'client_cert_path type should be string but is type int', 
    {
        "check_address" : None, 
        "check_port" : None, 
        "check_tls_config" : TypeError('client_cert_path type should be string but is type int')
    }),

    ({
        "address" : "localhost",
        "port" : 2**16,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" :"invalid_path",
            "client_cert_path" : PATH_VALID,
        }
    },
    {"check_address" : CallCount.ONE, "check_port" : CallCount.ONE, "check_tls_config" : CallCount.ONE},
    ValueError,  'invalid_path is not valid path to file', 
    {
        "check_address" : None, 
        "check_port" : None, 
        "check_tls_config" : ValueError('invalid_path is not valid path to file')
    }),
]

# (server_cert_path, client_cert_path, client_key_path,
# method_call_count_dict= {"method_name" : CallCount.NumberOfCalls})
CHANNEL_CERTS_VALID = [
    (SERVER_CERT_PATH_VALID, CLIENT_CERT_PATH_VALID, CLIENT_KEY_PATH_VALID,
    {"check_certificate_valid" : CallCount.TWO, "check_key_valid" : CallCount.ONE}),

    (SERVER_CERT_PATH_VALID, None, None,
    {"check_certificate_valid" : CallCount.ONE, "check_key_valid" : CallCount.ZERO})
]

# (certificate_path)
CERTIFICATE_VALID = [
    (PATH_VALID),
]

# (key_path)
PRIVATE_KEY_VALID = [
    (PATH_VALID),
]

# (config_dict,
# method_call_count_dict= {"method_name" : CallCount.NumberOfCalls})
BUILD_VALID = [
    ({
        "address" : "localhost",
        "port" : 9000
    }, {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ZERO}),
    ({
        "address" : "19.117.63.126",
        "port" : 1
    }, {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ZERO}),
    ({
        "address" : "cluster.cloud.iotg.intel.com",
        "port" : 2**16-1
    }, {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ZERO}),
    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "server_cert_path" : "valid_path"
        }
    }, {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ONE}),
    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "client_key_path" : PATH_VALID,
            "client_cert_path" : PATH_VALID,
            "server_cert_path" : PATH_VALID
        }
    }, {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ONE})
]

# (config_dict,
# expected_exception, expected_message,
# method_call_count_dict= {"method_name" : CallCount.NumberOfCalls})
BUILD_INVALID_CONFIG = [
    ({

    },
    ValueError, 'The minimal config must contain address and port',
    {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ZERO}),

    ({
        "address" : "localhost"
    },
    ValueError, 'The minimal config must contain address and port',
    {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ZERO}),

    ({
        "port" : 9000
    },
    ValueError, 'The minimal config must contain address and port',
    {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ZERO}),

    ({
        "address" : ["localhost"],
        "port" : 9000
    },
    TypeError, 'address type should be string, but is list',
    {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ZERO}),

    ({
        "address" : "address",
        "port" : '9000'
    },
    ValueError, 'address is not valid',
    {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ZERO}),

    ({
        "address" : "localhost",
        "port" : '9000'
    },
    TypeError, 'port type should be int, but is type str',
    {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ZERO}),

    ({
        "address" : "localhost",
        "port" : 2**16
    },
    ValueError, f"port should be in range <0, {2**16-1}>",
    {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ZERO}),
    

    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {

        }
    },
    ValueError, 'server_cert_path is not defined in tls_config',
    {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ZERO}),

    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" : PATH_VALID
        }
    },
    ValueError, 'none or both client_key_path and client_cert_path are required in tls_config',
    {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ZERO}),

    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" : PATH_VALID,
            "client_cert_path" : PATH_VALID,
            "invalid_key_name" : PATH_VALID
        }
    },
    ValueError,  'invalid_key_name is not valid tls_config key',
    {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ZERO}),

    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" :PATH_VALID,
            "client_cert_path" : 123,
        }
    },
    TypeError,  'client_cert_path type should be string but is type int',
    {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ZERO}),

    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" : "invalid_path",
            "client_cert_path" : PATH_VALID,
        }
    },
    ValueError,  'invalid_path is not valid path to file',
    {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ZERO}),

]

# (config_dict,
# expected_exception, expected_message,
# method_call_count_dict= {"method_name" : CallCount.NumberOfCalls})
BUILD_INVALID_CERTS = [
    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" : "path_to_invalid_private_key",
            "client_cert_path" : PATH_VALID,
        }
    },
    ValueError,  'path_to_invalid_private_key file is not valid private key',
    {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ONE}),

    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "server_cert_path" : "path_to_invalid_server_certificate",
            "client_key_path" : PATH_VALID,
            "client_cert_path" : PATH_VALID,
        }
    },
    ValueError,  'path_to_invalid_server_certificate is not valid certificate',
    {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ONE}),

    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "server_cert_path" : PATH_VALID,
            "client_key_path" : PATH_VALID,
            "client_cert_path" : "path_to_invalid_client_certificate",
        }
    },
    ValueError,  'path_to_invalid_client_certificate is not valid certificate',
    {"check_config" : CallCount.ONE, "prepare_certs" : CallCount.ONE}),
]

# response_dict = {
# 'version' : model_version,
# 'name' : model_name,
# 'inputs' : inputs_dict,
# 'outputs' : outputs_dict
# }
MODEL_METADATA_RESPONSE_VALID = [
    {
        'version' : 2,
        'name' : 'resnet',
        'inputs' : {
            '0' : {
                'shape' : [1, 3, 244, 244],
                'dtype' : DataType.DT_FLOAT
            }
        },
        'outputs' : {
            '1463' : {
                'shape' : [1, 1000],
                'dtype' : DataType.DT_FLOAT
            }
        }
    },

    {
        'version' : 1,
        'name' : 'model_name',
        'inputs' : {
            '0' : {
                'shape' : [1, 3, 244, 244],
                'dtype' : DataType.DT_FLOAT
            },
            '1' : {
                'shape' : [0, 1, 3, 244, 244],
                'dtype' : DataType.DT_INT32
            }
        },
        'outputs' : {
            '1463' : {
                'shape' : [1, 1000],
                'dtype' : DataType.DT_FLOAT
            },
            'second_output' : {
                'shape' : [0, 1, 1000],
                'dtype' : DataType.DT_INT32
            }
        }
    },

    {
        'version' : 1,
        'name' : 'model_name',
        'inputs' : {
            'input1' : {
                'shape' : [1, 3, 1080, 1920],
                'dtype' : DataType.DT_QINT32
            },
            'input2' : {
                'shape' : [1, 3, 244, 244],
                'dtype' : DataType.DT_INT32
            }
        },
        'outputs' : {
            'single_output' : {
                'shape' : [1, 7, 200 ,200],
                'dtype' : DataType.DT_FLOAT
            }
        }
    }
]

# inputs_dict,
# model_name, model_version, expected_exception, expected_message
PREDICT_REQUEST_INVALID_INPUTS = [
    ([],
    'model_name', 0, TypeError, "inputs type should be dict, but is list"),
    (('input1', [1,2,3]),
    'model_name', 0, TypeError, "inputs type should be dict, but is tuple"),
    ({
        1 : [1,2,3],
        "input2" : [1,2]
    }, 'model_name', 0, TypeError, "inputs keys should be type str, but found int"),
    ({
        "input1" : [[1.0, 2.0], [1.0, 2.0, 3.0]]
    }, 'model_name', 0, ValueError, "argument must be a dense tensor: [[1.0, 2.0], [1.0, 2.0, 3.0]] - got shape [2], but wanted [2, 2]"),
    ({
        "input1" : [[(1,2,3)],[(1,2)],[(1,2,3)]]
    }, 'model_name', 0, TypeError, "provided values type is not valid"),
    ({
        "input1" : float128(2.5)
    }, 'model_name', 0, TypeError, "provided values type is not valid"),
    ({
        "input1" : (1,2,3)
    }, 'model_name', 0, TypeError, "values type should be (list, np.ndarray, scalar), but is tuple"),
    ({
        "input1" : [
            [bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00]), bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00])],
            [bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00]), bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00])]
        ]
    }, 'model_name', 0, ValueError, "bytes values with dtype DT_STRING must be in shape [N]"),
]

# inputs_dict,
# expected_proto_dict,
# model_name, model_version
PREDICT_REQUEST_VALID = [
    ({
        "input1" : [1,2,3],
        "input2" : array([1.0,2.0,3.0]),
        "input3" : [[int32(3), int32(1)], [int32(4), int32(16)]],
    },
    {
        "input1" : {
            "field" : "tensor_content",
            "shape" : TensorShapeProto(dim = [TensorShapeProto.Dim(size=3)]),
            "dtype" : DataType.DT_INT64,
            'value' : array([1,2,3]).tobytes()
        },
        "input2" : {
            "field" : "tensor_content",
            "shape" : TensorShapeProto(dim = [TensorShapeProto.Dim(size=3)]),
            "dtype" : DataType.DT_DOUBLE,
            'value' : array([1.0,2.0,3.0]).tobytes()
        },
        "input3" : {
            "field" : "tensor_content",
            "shape" : TensorShapeProto(dim = [TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=2)]),
            "dtype" : DataType.DT_INT32,
            'value' : array([[int32(3), int32(1)], [int32(4), int32(16)]]).tobytes()
        },
    }, 'model_name', 0),
    
    ({
        "input1" : TensorProto(dtype=DataType.DT_INT8,
        tensor_shape=TensorShapeProto(dim= [TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
        tensor_content=array([1,2,3,4,5,6]).tobytes()),
        "input2" : 5.0,
        "input3" : bytes([1,2,3])
    },
    {
        "input2" : {
            "field" : "double_val",
            "shape" : TensorShapeProto(dim = [TensorShapeProto.Dim(size=1)]),
            "dtype" : DataType.DT_DOUBLE,
            'value' : array([5.0])
        },
        "input3" : {
            "field" : "string_val",
            "shape" : TensorShapeProto(dim = [TensorShapeProto.Dim(size=1)]),
            "dtype" : DataType.DT_STRING,
            'value' : [bytes([1,2,3])]
        }
    }, 'model_name', 0),

    ({

    },
    {

    }, 'model_name', 0)
]
PREDICT_RESPONSE_VALID = [
    ({
        "1463" : TensorProto(dtype=DataType.DT_INT8,
        tensor_shape=TensorShapeProto(dim= [TensorShapeProto.Dim(size=3)]),
        tensor_content=array([1,2,3], dtype=int8).tobytes()),
    }, "model_name", 0,
    {
        "1463" : array([1,2,3], dtype=int8)
    }),

    ({
        "1463" : TensorProto(dtype=DataType.DT_INT32,
        tensor_shape=TensorShapeProto(dim= [TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
        tensor_content=array([1,2,3,4,5,6], dtype=int32).tobytes()),
        "2" : TensorProto(dtype=DataType.DT_DOUBLE,
        tensor_shape=TensorShapeProto(dim= [TensorShapeProto.Dim(size=1)]),
        double_val=array([12.0], dtype=float64)),
    }, "model_name", 0,
    {
        "1463" : array([[1,2,3], [4,5,6]], dtype=int32),
        "2" : array([12.0], dtype=float64)
    }),

    ({
        "1463" : TensorProto(dtype=DataType.DT_STRING,
        tensor_shape=TensorShapeProto(dim= [TensorShapeProto.Dim(size=2)]),
        string_val=[bytes([1,2,3]), bytes([4,5])]),
        "2" : TensorProto(dtype=DataType.DT_STRING,
        tensor_shape=TensorShapeProto(dim= [TensorShapeProto.Dim(size=1)]),
        string_val=[bytes([1,2,3])]),
    }, "model_name", 0,
    {
        "1463" : [bytes([1,2,3]), bytes([4,5])],
        "2" : [bytes([1,2,3])]
    }),
]

PREDICT_RESPONSE_INVALID = [
    ({
        "1463" : TensorProto(),
    }, "model_name", 0, TypeError, "Unsupported tensor type: 0"),
    ({
        "1463" : TensorProto(dtype=DataType.DT_INVALID),
    }, "model_name", 0, TypeError, "Unsupported tensor type: 0"),
    ({
        "1463" : TensorProto(dtype=DataType.DT_RESOURCE),
    }, "model_name", 0, TypeError, "Unsupported tensor type: 20"),
]

# ({"model_name" : model_name, "model_version" : model_version, "model_raw_name": raw_request_model_name, "model_raw_version" : raw_request_model_version})
MODEL_STATUS_REQUEST_VALID = [
    ({"model_name" : "name", "model_version" : 0, "model_raw_name" : "name", "model_raw_version" : 0}),
]

# ({"model_name" : model_name, "model_version" : model_version, "model_raw_name": raw_request_model_name, "model_raw_version" : raw_request_model_version},
# expected_exception, expected_message)
MODEL_STATUS_REQUEST_INVALID_RAW_REQUEST = [
    ({"model_name" : "name", "model_version" : 0, "model_raw_name" : "other_name", "model_raw_version" : 0},
    ValueError, 'request is not valid GrpcModelStatusRequest'),
    ({"model_name" : "other_name", "model_version" : 0, "model_raw_name" : "name", "model_raw_version" : 0},
    ValueError, 'request is not valid GrpcModelStatusRequest'),
    ({"model_name" : "name", "model_version" : 0, "model_raw_name" : "name", "model_raw_version" : 1},
    ValueError, 'request is not valid GrpcModelStatusRequest'),
    ({"model_name" : "name", "model_version" : 1, "model_raw_name" : "name", "model_raw_version" : 0},
    ValueError, 'request is not valid GrpcModelStatusRequest'),
]

# (request, expeceted_exception, expected_message)
MODEL_STATUS_REQUEST_INVALID_REQUEST_TYPE = [
    (None, TypeError, "request type should be GrpcModelStatusRequest, but is NoneType"),
    (GetModelStatusRequest(), TypeError, "request type should be GrpcModelStatusRequest, but is GetModelStatusRequest"),
]

# ({"model_name" : model_name, "model_version" : model_version, "model_raw_name": raw_request_model_name, "model_raw_version" : raw_request_model_version},
# expected_message, grpc_error_status_code, grpc_error_details)
GET_MODEL_STATUS_INVALID_GRPC = [
    (f"There was an error during sending ModelStatusRequest. Grpc exited with: \n{StatusCode.UNAVAILABLE.name} - failed to connect to all adresses",
    StatusCode.UNAVAILABLE, "failed to connect to all adresses"),
    (f"There was an error during sending ModelStatusRequest. Grpc exited with: \n{StatusCode.UNAVAILABLE.name} - Empty update",
    StatusCode.UNAVAILABLE, "Empty update"),
    (f"There was an error during sending ModelStatusRequest. Grpc exited with: \n{StatusCode.NOT_FOUND.name} - Model with requested version is not found",
    StatusCode.NOT_FOUND, "Model with requested version is not found"),
    (f"There was an error during sending ModelStatusRequest. Grpc exited with: \n{StatusCode.NOT_FOUND.name} - Model with requested name is not found",
    StatusCode.NOT_FOUND, "Model with requested name is not found"),
]

# ({"model_name" : model_name, "model_version" : model_version, "model_raw_name": raw_request_model_name, "model_raw_version" : raw_request_model_version, 
# "metadata_field_list" : raw_request_metadata_fields})
MODEL_METADATA_REQUEST_VALID = [
    ({"model_name" : "name", "model_version" : 0, "model_raw_name" : "name", "model_raw_version" : 0, "metadata_field_list" : ["signature_def"]}),
]

# ({"model_name" : model_name, "model_version" : model_version, "model_raw_name": raw_request_model_name, "model_raw_version" : raw_request_model_version, 
# "metadata_field_list" : raw_request_metadata_fields},
# expected_exception, expected_message)
MODEL_METADATA_REQUEST_INVALID_RAW_REQUEST = [
    ({"model_name" : "name", "model_version" : 0, "model_raw_name" : "other_name", "model_raw_version" : 0, "metadata_field_list" : ["signature_def"]},
    ValueError, 'request is not valid GrpcModelStatusRequest'),
    ({"model_name" : "other_name", "model_version" : 0, "model_raw_name" : "name", "model_raw_version" : 0, "metadata_field_list" : ["signature_def"]},
    ValueError, 'request is not valid GrpcModelStatusRequest'),
    ({"model_name" : "name", "model_version" : 0, "model_raw_name" : "name", "model_raw_version" : 1, "metadata_field_list" : ["signature_def"]},
    ValueError, 'request is not valid GrpcModelStatusRequest'),
    ({"model_name" : "name", "model_version" : 1, "model_raw_name" : "name", "model_raw_version" : 0, "metadata_field_list" : ["signature_def"]},
    ValueError, 'request is not valid GrpcModelStatusRequest'),
    ({"model_name" : "name", "model_version" : 1, "model_raw_name" : "name", "model_raw_version" : 1, "metadata_field_list" : ["invalid"]},
    ValueError, 'request is not valid GrpcModelStatusRequest'),
]

# (request, expected_exception, expected_message)
MODEL_METADATA_REQUEST_INVALID_REQUEST_TYPE = [
    (None, TypeError, "request type should be GrpcModelMetadataRequest, but is NoneType"),
    (GetModelMetadataRequest(), TypeError, "request type should be GrpcModelMetadataRequest, but is GetModelMetadataRequest"),
]

# expected_message, grpc_error_status_code, grpc_error_details)
GET_MODEL_METADATA_INVALID_GRPC = [
    (f"There was an error during sending ModelStatusRequest. Grpc exited with: \n{StatusCode.UNAVAILABLE.name} - failed to connect to all adresses",
    StatusCode.UNAVAILABLE, "failed to connect to all adresses"),
    (f"There was an error during sending ModelStatusRequest. Grpc exited with: \n{StatusCode.UNAVAILABLE.name} - Empty update",
    StatusCode.UNAVAILABLE, "Empty update"),
    (f"There was an error during sending ModelStatusRequest. Grpc exited with: \n{StatusCode.NOT_FOUND.name} - Model with requested version is not found",
    StatusCode.NOT_FOUND, "Model with requested version is not found"),
    (f"There was an error during sending ModelStatusRequest. Grpc exited with: \n{StatusCode.NOT_FOUND.name} - Model with requested name is not found",
    StatusCode.NOT_FOUND, "Model with requested name is not found"),
]
