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

from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusResponse, ModelVersionStatus
from tensorflow.core.protobuf.error_codes_pb2 import Code
from tensorflow_serving.util.status_pb2 import StatusProto

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

def create_model_status_response(model_version, error_code, error_message, model_state):
    status = StatusProto()
    status.error_code = error_code
    status.error_message = error_message

    model_version_status = ModelVersionStatus()
    model_version_status.version = model_version
    model_version_status.state = model_state
    model_version_status.status.CopyFrom(status)

    return model_version_status

def merge_model_status_responses(responses):
    raw_response = GetModelStatusResponse()
    model_versions = [response for response in responses]
    raw_response.model_version_status.extend(model_versions)

    return raw_response

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

PATH_VALID = "valid_path"

TLS_CONFIG_VALID = [
    ({"server_cert_path" : PATH_VALID}, 1),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : PATH_VALID, "server_cert_path" : PATH_VALID}, 3),
]

TLS_CONFIG_INVALID = [
    ({"client_key_path" : PATH_VALID, "server_cert_path" : PATH_VALID,} , ValueError, "none or both client_key_path and client_cert_path are required in tls_config", 0),
    ({"client_cert_path" : PATH_VALID, "server_cert_path" : PATH_VALID}, ValueError, "none or both client_key_path and client_cert_path are required in tls_config", 0),
    
    ({"non_client_key_path" : PATH_VALID, "client_cert_path" : PATH_VALID, "server_cert_path" : PATH_VALID}, ValueError, "non_client_key_path is not valid tls_config key", 0),
    ({"client_key_path" : PATH_VALID, "non_client_cert_path" : PATH_VALID, "server_cert_path" : PATH_VALID}, ValueError, "non_client_cert_path is not valid tls_config key", 0),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : PATH_VALID, "non_server_cert_path" : PATH_VALID}, ValueError, "server_cert_path is not defined in tls_config", 0),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : PATH_VALID}, ValueError, "server_cert_path is not defined in tls_config", 0),

    ({"client_key_path" : 1, "client_cert_path" : PATH_VALID, "server_cert_path" : PATH_VALID}, TypeError, f'client_key_path type should be string but is type int', 0),
    ({"client_key_path" : [PATH_VALID], "client_cert_path" : PATH_VALID, "server_cert_path" : PATH_VALID}, TypeError, f'client_key_path type should be string but is type list', 0),
    ({"client_key_path" : 1, "client_cert_path" : PATH_VALID, "server_cert_path" : 1}, TypeError, f'client_key_path type should be string but is type int', 0),

    ({"client_key_path" : PATH_VALID, "client_cert_path" : 1, "server_cert_path" : PATH_VALID}, TypeError, f'client_cert_path type should be string but is type int', 1),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : 1, "server_cert_path" : 1}, TypeError, f'client_cert_path type should be string but is type int', 1),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : [PATH_VALID], "server_cert_path" : PATH_VALID}, TypeError, f'client_cert_path type should be string but is type list', 1),

    ({"client_key_path" : PATH_VALID, "client_cert_path" : PATH_VALID, "server_cert_path" : 1}, TypeError, f'server_cert_path type should be string but is type int', 2),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : 1, "server_cert_path" : [PATH_VALID]}, TypeError, f'client_cert_path type should be string but is type int', 1),
    
    ({"client_key_path" : "invalid", "client_cert_path" : PATH_VALID, "server_cert_path" : PATH_VALID}, ValueError, f'invalid is not valid path to file', 1),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : "/very/invalid", "server_cert_path" : PATH_VALID}, ValueError, f'/very/invalid is not valid path to file', 2),
    ({"client_key_path" : PATH_VALID, "client_cert_path" : PATH_VALID, "server_cert_path" : "third_invalid_path"}, ValueError, f'third_invalid_path is not valid path to file', 3),

]

CONFIG_VALID = [
    ({
        "address" : "localhost",
        "port" : 9000
    }, {"check_address" : 1, "check_port" : 1, "check_tls_config" : 0}),
    ({
        "address" : "19.117.63.126",
        "port" : 1
    }, {"check_address" : 1, "check_port" : 1, "check_tls_config" : 0}),
    ({
        "address" : "cluster.cloud.iotg.intel.com",
        "port" : 2**16-1
    }, {"check_address" : 1, "check_port" : 1, "check_tls_config" : 0}),
    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "server_cert_path" : "valid_path"
        }
    }, {"check_address" : 1, "check_port" : 1, "check_tls_config" : 1}),
    ({
        "address" : "localhost",
        "port" : 9000,
        "tls_config" : {
            "client_key_path" : "valid_path",
            "client_cert_path" : "valid_path",
            "server_cert_path" : "valid_path"
        }
    }, {"check_address" : 1, "check_port" : 1, "check_tls_config" : 1})
]

CONFIG_INVALID = [
    ({

    },
    {"check_address" : 0, "check_port" : 0, "check_tls_config" : 0},
    Exception, 'The minimal config must contain address and port', 
    {"check_address" : None, "check_port" : None, "check_tls_config" : None}),
    ({
        "address" : "localhost"
    },
    {"check_address" : 0, "check_port" : 0, "check_tls_config" : 0},
    Exception, 'The minimal config must contain address and port', 
    {"check_address" : None, "check_port" : None, "check_tls_config" : None}),
    ({
        "port" : 9000
    },
    {"check_address" : 0, "check_port" : 0, "check_tls_config" : 0},
    Exception, 'The minimal config must contain address and port', 
    {"check_address" : None, "check_port" : None, "check_tls_config" : None}),

    ({
        "address" : ["localhost"],
        "port" : 9000
    },
    {"check_address" : 1, "check_port" : 0, "check_tls_config" : 0},
    TypeError, 'address type should be string, but is list', 
    {"check_address" : TypeError('address type should be string, but is list'), "check_port" : None, "check_tls_config" : None}),
    ({
        "address" : "address",
        "port" : '9000'
    },
    {"check_address" : 1, "check_port" : 0, "check_tls_config" : 0},
    ValueError, 'address is not valid', 
    {"check_address" : ValueError('address is not valid'), "check_port" : None, "check_tls_config" : None}),
    ({
        "address" : "localhost",
        "port" : '9000'
    },
    {"check_address" : 1, "check_port" : 1, "check_tls_config" : 0},
    TypeError, 'port type should be int, but is type str', 
    {"check_address" : None, "check_port" : TypeError('port type should be int, but is type str'), "check_tls_config" : None}),
    ({
        "address" : "localhost",
        "port" : 2**16
    },
    {"check_address" : 1, "check_port" : 1, "check_tls_config" : 0},
    ValueError, f"port should be in range <0, {2**16-1}>", 
    {"check_address" : None, "check_port" : ValueError(f"port should be in range <0, {2**16-1}>"), "check_tls_config" : None}),

    ({
        "address" : "localhost",
        "port" : 2**16,
        "tls_config" : {

        }
    },
    {"check_address" : 1, "check_port" : 1, "check_tls_config" : 1},
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
    {"check_address" : 1, "check_port" : 1, "check_tls_config" : 1},
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
    {"check_address" : 1, "check_port" : 1, "check_tls_config" : 1},
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
    {"check_address" : 1, "check_port" : 1, "check_tls_config" : 1},
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
    {"check_address" : 1, "check_port" : 1, "check_tls_config" : 1},
    ValueError,  'invalid_path is not valid path to file', 
    {"check_address" : None, "check_port" : None, "check_tls_config" : ValueError('invalid_path is not valid path to file')}),
]

SERVER_CERT_PATH_VALID, SERVER_CERT_PATH_INVALID = "server_cert_path_valid", "server_cert_path_invalid"
CLIENT_CERT_PATH_VALID, CLIENT_CERT_PATH_INVALID = "client_cert_path_valid", "client_cert_path_invalid"
CLIENT_KEY_PATH_VALID, CLIENT_KEY_PATH_INVALID = "client_key_path_valid", "client_key_path_invalid"

CHANNEL_CERTS_VALID = [
    (CLIENT_KEY_PATH_VALID, CLIENT_CERT_PATH_VALID, SERVER_CERT_PATH_VALID, {"load_privatekey" : 1, "load_certificate" : 2, "open" : 3}),
    (None, None, SERVER_CERT_PATH_VALID, {"load_privatekey" : 0, "load_certificate" : 1, "open" : 1})
]

CHANNEL_CERTS_INVALID = [
    (None, None, SERVER_CERT_PATH_INVALID, {"load_privatekey" : 0, "load_certificate" : 1, "open" : 1},
    ValueError, 'server certificate file is not valid',
    {"load_privatekey" : None, "load_certificate" : ValueError('server certificate file is not valid')}),
    (CLIENT_KEY_PATH_VALID, CLIENT_CERT_PATH_VALID, SERVER_CERT_PATH_INVALID, {"load_privatekey" : 1, "load_certificate" : 2, "open" : 3},
    ValueError, 'server certificate file is not valid',
    {"load_privatekey" : None, "load_certificate" : [None, ValueError('server certificate file is not valid')]}),
    (CLIENT_KEY_PATH_INVALID, CLIENT_CERT_PATH_VALID, SERVER_CERT_PATH_INVALID, {"load_privatekey" : 1, "load_certificate" : 0, "open" : 1},
    ValueError, 'client key file is not valid',
    {"load_privatekey" : ValueError('client key file is not valid'), "load_certificate" : None}),
    (CLIENT_KEY_PATH_INVALID, CLIENT_CERT_PATH_VALID, SERVER_CERT_PATH_VALID, {"load_privatekey" : 1, "load_certificate" : 0, "open" : 1},
    ValueError, 'client key file is not valid',
    {"load_privatekey" : ValueError('client key file is not valid'), "load_certificate" : None}),
    (CLIENT_KEY_PATH_VALID, CLIENT_CERT_PATH_INVALID, SERVER_CERT_PATH_INVALID, {"load_privatekey" : 1, "load_certificate" : 1, "open" : 2},
    ValueError, 'client certificate file is not valid',
    {"load_privatekey" : None, "load_certificate" : ValueError('client certificate file is not valid')}),
    (CLIENT_KEY_PATH_VALID, CLIENT_CERT_PATH_INVALID, SERVER_CERT_PATH_VALID, {"load_privatekey" : 1, "load_certificate" : 1, "open" : 2},
    ValueError, 'client certificate file is not valid',
    {"load_privatekey" : None, "load_certificate" : ValueError('client certificate file is not valid')}),
]
