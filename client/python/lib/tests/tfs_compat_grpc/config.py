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

from grpc import StatusCode
from numpy import array, float64, int32, int8, float128, float32

from config import CallCount, PATH_VALID # noqa

from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from tensorflow_serving.apis.get_model_status_pb2 import ModelVersionStatus
from tensorflow.core.framework.types_pb2 import DataType
from tensorflow.core.protobuf.error_codes_pb2 import Code
from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusRequest
from tensorflow_serving.apis.get_model_metadata_pb2 import GetModelMetadataRequest
from tensorflow_serving.apis.predict_pb2 import PredictRequest

from ovmsclient.tfs_compat.grpc.requests import (GrpcModelMetadataRequest, GrpcModelStatusRequest,
                                                 GrpcPredictRequest)

# responses_dict = {
#    model_version: { expected_status }
# }
MODEL_STATUS_RESPONSE_VALID = [{
    1: {"state": ModelVersionStatus.State.AVAILABLE, "error_code": Code.OK, "error_message": ""}
}, {
    2: {"state": ModelVersionStatus.State.END, "error_code": Code.OK, "error_message": ""},
    3: {"state": ModelVersionStatus.State.AVAILABLE, "error_code": Code.OK, "error_message": ""}
}, {
    1: {"state": ModelVersionStatus.State.START, "error_code": Code.OK, "error_message": ""},
    2: {"state": ModelVersionStatus.State.LOADING, "error_code": Code.UNKNOWN,
        "error_message": "Could not load CNN"},
    3: {"state": ModelVersionStatus.State.UNLOADING, "error_code": Code.OK, "error_message": ""}
}
]


# response_dict = {
# 'version': model_version,
# 'name': model_name,
# 'inputs': inputs_dict,
# 'outputs': outputs_dict
# }
MODEL_METADATA_RESPONSE_VALID = [
    {
        'version': 2,
        'name': 'resnet',
        'inputs': {
            '0': {
                'shape': [1, 3, 244, 244],
                'dtype': DataType.DT_FLOAT
            }
        },
        'outputs': {
            '1463': {
                'shape': [1, 1000],
                'dtype': DataType.DT_FLOAT
            }
        }
    },

    {
        'version': 1,
        'name': 'model_name',
        'inputs': {
            '0': {
                'shape': [1, 3, 244, 244],
                'dtype': DataType.DT_FLOAT
            },
            '1': {
                'shape': [0, 1, 3, 244, 244],
                'dtype': DataType.DT_INT32
            }
        },
        'outputs': {
            '1463': {
                'shape': [1, 1000],
                'dtype': DataType.DT_FLOAT
            },
            'second_output': {
                'shape': [0, 1, 1000],
                'dtype': DataType.DT_INT32
            }
        }
    },

    {
        'version': 1,
        'name': 'model_name',
        'inputs': {
            'input1': {
                'shape': [1, 3, 1080, 1920],
                'dtype': DataType.DT_QINT32
            },
            'input2': {
                'shape': [1, 3, 244, 244],
                'dtype': DataType.DT_INT32
            }
        },
        'outputs': {
            'single_output': {
                'shape': [1, 7, 200, 200],
                'dtype': DataType.DT_FLOAT
            }
        }
    }
]

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
# expected_proto_dict,
# model_name, model_version)
PREDICT_REQUEST_VALID = [
    ({
        "input1": [1, 2, 3],
        "input2": array([1.0, 2.0, 3.0]),
        "input3": [[int32(3), int32(1)], [int32(4), int32(16)]],
    }, {
        "input1": {
            "field": "tensor_content",
            "shape": TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]),
            "dtype": DataType.DT_INT32,
            'value': array([1, 2, 3], dtype=int32).tobytes()
        },
        "input2": {
            "field": "tensor_content",
            "shape": TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]),
            "dtype": DataType.DT_DOUBLE,
            'value': array([1.0, 2.0, 3.0]).tobytes()
        },
        "input3": {
            "field": "tensor_content",
            "shape": TensorShapeProto(dim=[TensorShapeProto.Dim(size=2),
                                           TensorShapeProto.Dim(size=2)]),
            "dtype": DataType.DT_INT32,
            'value': array([[int32(3), int32(1)], [int32(4), int32(16)]]).tobytes()
        },
    }, 'model_name', 0),

    ({
        "input1": TensorProto(dtype=DataType.DT_INT8,
                              tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=2),
                                                            TensorShapeProto.Dim(size=3)]),
                              tensor_content=array([1, 2, 3, 4, 5, 6]).tobytes()),
        "input2": 5.0,
        "input3": bytes([1, 2, 3])
    }, {
        "input2": {
            "field": "float_val",
            "shape": TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
            "dtype": DataType.DT_FLOAT,
            'value': array([5.0], dtype=float32)
        },
        "input3": {
            "field": "string_val",
            "shape": TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
            "dtype": DataType.DT_STRING,
            'value': [bytes([1, 2, 3])]
        }
    }, 'model_name', 0),

    ({

    }, {

    }, 'model_name', 0)
]

# (response_outputs_dict, model_name, model_version, expected_outputs_dict)
PREDICT_RESPONSE_VALID = [
    ({
        "1463": TensorProto(dtype=DataType.DT_INT8,
                            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]),
                            tensor_content=array([1, 2, 3], dtype=int8).tobytes()),
    }, "model_name", 0,
     {
        "1463": array([1, 2, 3], dtype=int8)
    }),

    ({
        "1463": TensorProto(dtype=DataType.DT_INT32,
                            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=2),
                                                          TensorShapeProto.Dim(size=3)]),
                            tensor_content=array([1, 2, 3, 4, 5, 6], dtype=int32).tobytes()),
        "2": TensorProto(dtype=DataType.DT_DOUBLE,
                         tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
                         double_val=array([12.0], dtype=float64)),
    }, "model_name", 0,
     {
        "1463": array([[1, 2, 3], [4, 5, 6]], dtype=int32),
        "2": array([12.0], dtype=float64)
    }),

    ({
        "1463": TensorProto(dtype=DataType.DT_STRING,
                            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=2)]),
                            string_val=[bytes([1, 2, 3]), bytes([4, 5])]),
        "2": TensorProto(dtype=DataType.DT_STRING,
                         tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
                         string_val=[bytes([1, 2, 3])]),
    }, "model_name", 0,
     {
        "1463": [bytes([1, 2, 3]), bytes([4, 5])],
        "2": [bytes([1, 2, 3])]
    }),
]

# (response_outputs_dict, model_name, model_version, expected_exception, expected_message)
PREDICT_RESPONSE_INVALID = [
    ({
        "1463": TensorProto(),
    }, "model_name", 0, TypeError, "Unsupported tensor type: 0"),
    ({
        "1463": TensorProto(dtype=DataType.DT_INVALID),
    }, "model_name", 0, TypeError, "Unsupported tensor type: 0"),
    ({
        "1463": TensorProto(dtype=DataType.DT_RESOURCE),
    }, "model_name", 0, TypeError, "Unsupported tensor type: 20"),
]

# ({"model_name": model_name, "model_version": model_version,
#   "raw_request_model_name": raw_request_model_name, "raw_request_model_version": raw_request_model_version})# noqa : E501
MODEL_STATUS_REQUEST_VALID = [
    ({"model_name": "name", "model_version": 0,
      "raw_request_model_name": "name", "raw_request_model_version": 0}),
]

# ({"model_name": model_name, "model_version": model_version,
#   "raw_request_model_name": raw_request_model_name, "raw_request_model_version": raw_request_model_version},# noqa : E501
# expected_exception, expected_message)
MODEL_STATUS_REQUEST_INVALID_RAW_REQUEST = [
    ({"model_name": "name", "model_version": 0,
      "raw_request_model_name": "other_name", "raw_request_model_version": 0},
     ValueError, 'request is not valid GrpcModelStatusRequest'),
    ({"model_name": "other_name", "model_version": 0,
      "raw_request_model_name": "name", "raw_request_model_version": 0},
     ValueError, 'request is not valid GrpcModelStatusRequest'),
    ({"model_name": "name", "model_version": 0,
      "raw_request_model_name": "name", "raw_request_model_version": 1},
     ValueError, 'request is not valid GrpcModelStatusRequest'),
    ({"model_name": "name", "model_version": 1,
      "raw_request_model_name": "name", "raw_request_model_version": 0},
     ValueError, 'request is not valid GrpcModelStatusRequest'),
]

# (request, expeceted_exception, expected_message)
MODEL_STATUS_REQUEST_INVALID_REQUEST_TYPE = [
    (None, TypeError,
     "request type should be GrpcModelStatusRequest, but is NoneType"),
    (GetModelStatusRequest(), TypeError,
     "request type should be GrpcModelStatusRequest, but is GetModelStatusRequest"),
    (GrpcModelStatusRequest('model_name', 0, 'raw_request'), TypeError,
     "request is not valid GrpcModelStatusRequest")
]


# (expected_message, grpc_error_status_code, grpc_error_details)
GET_MODEL_STATUS_INVALID_GRPC = [
    ("There was an error during sending ModelStatusRequest. "
     f"Grpc exited with: \n{StatusCode.UNAVAILABLE.name} - failed to connect to all adresses",
     StatusCode.UNAVAILABLE, "failed to connect to all adresses"),
    ("There was an error during sending ModelStatusRequest. "
     f"Grpc exited with: \n{StatusCode.UNAVAILABLE.name} - Empty update",
     StatusCode.UNAVAILABLE, "Empty update"),
    ("There was an error during sending ModelStatusRequest. "
     f"Grpc exited with: \n{StatusCode.NOT_FOUND.name} - Model with requested version is not found",
     StatusCode.NOT_FOUND, "Model with requested version is not found"),
    ("There was an error during sending ModelStatusRequest. "
     f"Grpc exited with: \n{StatusCode.NOT_FOUND.name} - Model with requested name is not found",
     StatusCode.NOT_FOUND, "Model with requested name is not found"),
]

# ({"model_name": model_name, "model_version": model_version,
#   "raw_request_model_name": raw_request_model_name, "raw_request_model_version": raw_request_model_version,# noqa : E501
# "metadata_field_list": raw_request_metadata_fields})
MODEL_METADATA_REQUEST_VALID = [
    ({"model_name": "name", "model_version": 0,
      "raw_request_model_name": "name", "raw_request_model_version": 0,
      "metadata_field_list": ["signature_def"]}),
]

# ({"model_name": model_name, "model_version": model_version,
#   "raw_request_model_name": raw_request_model_name, "raw_request_model_version": raw_request_model_version,# noqa : E501
# "metadata_field_list": raw_request_metadata_fields},
# expected_exception, expected_message)
MODEL_METADATA_REQUEST_INVALID_RAW_REQUEST = [
    ({"model_name": "name", "model_version": 0,
      "raw_request_model_name": "other_name", "raw_request_model_version": 0,
      "metadata_field_list": ["signature_def"]},
     ValueError, 'request is not valid GrpcModelMetadataRequest'),
    ({"model_name": "other_name", "model_version": 0,
      "raw_request_model_name": "name", "raw_request_model_version": 0,
      "metadata_field_list": ["signature_def"]},
     ValueError, 'request is not valid GrpcModelMetadataRequest'),
    ({"model_name": "name", "model_version": 0,
      "raw_request_model_name": "name", "raw_request_model_version": 1,
      "metadata_field_list": ["signature_def"]},
     ValueError, 'request is not valid GrpcModelMetadataRequest'),
    ({"model_name": "name", "model_version": 1,
      "raw_request_model_name": "name", "raw_request_model_version": 0,
      "metadata_field_list": ["signature_def"]},
     ValueError, 'request is not valid GrpcModelMetadataRequest'),
    ({"model_name": "name", "model_version": 1,
      "raw_request_model_name": "name", "raw_request_model_version": 1,
      "metadata_field_list": ["invalid"]},
     ValueError, 'request is not valid GrpcModelMetadataRequest'),
]

# (request, expected_exception, expected_message)
MODEL_METADATA_REQUEST_INVALID_REQUEST_TYPE = [
    (None, TypeError,
     "request type should be GrpcModelMetadataRequest, but is NoneType"),
    (GetModelMetadataRequest(), TypeError,
     "request type should be GrpcModelMetadataRequest, but is GetModelMetadataRequest"),
    (GrpcModelMetadataRequest('model_name', 0, 'raw_request'), TypeError,
     "request is not valid GrpcModelMetadataRequest")
]

# expected_message, grpc_error_status_code, grpc_error_details)
GET_MODEL_METADATA_INVALID_GRPC = [
    ("There was an error during sending ModelMetadataRequest. "
     f"Grpc exited with: \n{StatusCode.UNAVAILABLE.name} - failed to connect to all adresses",
     StatusCode.UNAVAILABLE, "failed to connect to all adresses"),
    ("There was an error during sending ModelMetadataRequest. "
     f"Grpc exited with: \n{StatusCode.UNAVAILABLE.name} - Empty update",
     StatusCode.UNAVAILABLE, "Empty update"),
    ("There was an error during sending ModelMetadataRequest. "
     f"Grpc exited with: \n{StatusCode.NOT_FOUND.name} - Model with requested version is not found",
     StatusCode.NOT_FOUND, "Model with requested version is not found"),
    ("There was an error during sending ModelMetadataRequest. "
     f"Grpc exited with: \n{StatusCode.NOT_FOUND.name} - Model with requested name is not found",
     StatusCode.NOT_FOUND, "Model with requested name is not found"),
]

# ({"model_name": model_name, "model_version": model_version,
#   "raw_request_model_name": raw_request_model_name, "raw_request_model_version": raw_request_model_version,# noqa : E501
# "inputs_dict": inputs_for_request, "raw_request_inputs_dict": inputs_for_raw_request})
PREDICT_REQUEST_VALID_SPEC = [
    ({"model_name": "name", "model_version": 0,
      "raw_request_model_name": "name", "raw_request_model_version": 0,
      "inputs_dict": {
          "0": TensorProto(dtype=DataType.DT_INT8,
                           tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]),
                           tensor_content=array([1, 2, 3]).tobytes())
      },
      "raw_request_inputs_dict": {
          "0": TensorProto(dtype=DataType.DT_INT8,
                           tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]),
                           tensor_content=array([1, 2, 3]).tobytes())
      }}),
]

# ({"model_name": model_name, "model_version": model_version,
#   "raw_request_model_name": raw_request_model_name, "raw_request_model_version": raw_request_model_version,# noqa : E501
# "inputs_dict": inputs_for_request, "raw_request_inputs_dict": inputs_for_raw_request},
# expected_exception, expected_message)
PREDICT_REQUEST_INVALID_SPEC_RAW_REQUEST = [
    ({"model_name": "name", "model_version": 0,
      "raw_request_model_name": "other_name", "raw_request_model_version": 0,
      "inputs_dict": {
          "0": TensorProto()
      },
      "raw_request_inputs_dict": {
          "0": TensorProto()
      }}, ValueError, 'request is not valid GrpcPredictRequest'),
    ({"model_name": "other_name", "model_version": 0,
      "raw_request_model_name": "name", "raw_request_model_version": 0,
      "inputs_dict": {
          "0": TensorProto()
      },
      "raw_request_inputs_dict": {
          "0": TensorProto()
      }}, ValueError, 'request is not valid GrpcPredictRequest'),
    ({"model_name": "name", "model_version": 1,
      "raw_request_model_name": "name", "raw_request_model_version": 0,
      "inputs_dict": {
          "0": TensorProto()
      },
      "raw_request_inputs_dict": {
          "0": TensorProto()
      }}, ValueError, 'request is not valid GrpcPredictRequest'),
    ({"model_name": "name", "model_version": 0,
      "raw_request_model_name": "name", "raw_request_model_version": 1,
      "inputs_dict": {
          "0": TensorProto()
      },
      "raw_request_inputs_dict": {
          "0": TensorProto()
      }}, ValueError, 'request is not valid GrpcPredictRequest'),
    ({"model_name": "name", "model_version": 0,
      "raw_request_model_name": "name", "raw_request_model_version": 0,
      "inputs_dict": {
          "0": TensorProto()
      },
      "raw_request_inputs_dict": {
          "1": TensorProto()
      }}, ValueError, 'request is not valid GrpcPredictRequest'),
]

# (predict_request, expected_exception, expected_message)
PREDICT_REQUEST_INVALID_SPEC_TYPE = [
    (None, TypeError,
     'request type should be GrpcPredictRequest, but is NoneType'),
    (PredictRequest(), TypeError,
     'request type should be GrpcPredictRequest, but is PredictRequest'),
    (GrpcPredictRequest({}, "model_name", 0, "raw_request"),
     TypeError, 'request is not valid GrpcPredictRequest'),
]

# (expected_message, grpc_error_status_code, grpc_error_details)
PREDICT_INVAlID_GRPC = [
    ("There was an error during sending PredictRequest. "
     f"Grpc exited with: \n{StatusCode.UNAVAILABLE.name} - "
     "failed to connect to all adresses",
     StatusCode.UNAVAILABLE, "failed to connect to all adresses"),
    ("There was an error during sending PredictRequest. "
     f"Grpc exited with: \n{StatusCode.NOT_FOUND.name} - "
     "Model with requested version is not found",
     StatusCode.NOT_FOUND, "Model with requested version is not found"),
    ("There was an error during sending PredictRequest. "
     f"Grpc exited with: \n{StatusCode.NOT_FOUND.name} - "
     "Model with requested name is not found",
     StatusCode.NOT_FOUND, "Model with requested name is not found"),
    ("There was an error during sending PredictRequest. "
     f"Grpc exited with: \n{StatusCode.INVALID_ARGUMENT.name} - "
     "Invalid input precision - Expected: FP32; Actual: I64",
     StatusCode.INVALID_ARGUMENT, ("Invalid input precision - "
                                   "Expected: FP32; Actual: I64")),
    ("There was an error during sending PredictRequest. "
     f"Grpc exited with: \n{StatusCode.INVALID_ARGUMENT.name} - "
     "Invalid number of inputs - Expected: 1; Actual: 0",
     StatusCode.INVALID_ARGUMENT, ("Invalid number of inputs - "
                                   "Expected: 1; Actual: 0")),
    ("There was an error during sending PredictRequest. "
     f"Grpc exited with: \n{StatusCode.INVALID_ARGUMENT.name} - "
     "Missing input with specific name - Required input: 0",
     StatusCode.INVALID_ARGUMENT, ("Missing input with specific name - "
                                   "Required input: 0")),
    ("There was an error during sending PredictRequest. "
     f"Grpc exited with: \n{StatusCode.INVALID_ARGUMENT.name} - "
     "Invalid number of shape dimensions - Expected: (1,3,224,224); Actual: (3)",
     StatusCode.INVALID_ARGUMENT, ("Invalid number of shape dimensions - "
                                   "Expected: (1,3,224,224); Actual: (3)")),
]

# (config_dict,
# method_call_count_dict= {"method_name": CallCount.NumberOfCalls})
BUILD_VALID = [
    ({
        "address": "localhost",
        "port": 9000
    }, {"check_config": CallCount.ONE, "prepare_certs": CallCount.ZERO}),
    ({
        "address": "19.117.63.126",
        "port": 1
    }, {"check_config": CallCount.ONE, "prepare_certs": CallCount.ZERO}),
    ({
        "address": "cluster.cloud.iotg.intel.com",
        "port": 2**16-1
    }, {"check_config": CallCount.ONE, "prepare_certs": CallCount.ZERO}),
    ({
        "address": "localhost",
        "port": 9000,
        "tls_config": {
            "server_cert_path": "valid_path"
        }
    }, {"check_config": CallCount.ONE, "prepare_certs": CallCount.ONE}),
    ({
        "address": "localhost",
        "port": 9000,
        "tls_config": {
            "client_key_path": PATH_VALID,
            "client_cert_path": PATH_VALID,
            "server_cert_path": PATH_VALID
        }
    }, {"check_config": CallCount.ONE, "prepare_certs": CallCount.ONE})
]

# (config_dict,
# expected_exception, expected_message,
# method_call_count_dict= {"method_name": CallCount.NumberOfCalls})
BUILD_INVALID_CONFIG = [
    ({

    },
     ValueError, 'The minimal config must contain address and port',
     {"check_config": CallCount.ONE, "prepare_certs": CallCount.ZERO}),

    ({
        "address": "localhost"
    },
     ValueError, 'The minimal config must contain address and port',
     {"check_config": CallCount.ONE, "prepare_certs": CallCount.ZERO}),

    ({
        "port": 9000
    },
     ValueError, 'The minimal config must contain address and port',
     {"check_config": CallCount.ONE, "prepare_certs": CallCount.ZERO}),

    ({
        "address": ["localhost"],
        "port": 9000
    },
     TypeError, 'address type should be string, but is list',
     {"check_config": CallCount.ONE, "prepare_certs": CallCount.ZERO}),

    ({
        "address": "address",
        "port": '9000'
    },
     ValueError, 'address is not valid',
     {"check_config": CallCount.ONE, "prepare_certs": CallCount.ZERO}),

    ({
        "address": "localhost",
        "port": '9000'
    },
     TypeError, 'port type should be int, but is type str',
     {"check_config": CallCount.ONE, "prepare_certs": CallCount.ZERO}),

    ({
        "address": "localhost",
        "port": 2**16
    },
     ValueError, f"port should be in range <0, {2**16-1}>",
     {"check_config": CallCount.ONE, "prepare_certs": CallCount.ZERO}),


    ({
        "address": "localhost",
        "port": 9000,
        "tls_config": {

        }
    },
     ValueError, 'server_cert_path is not defined in tls_config',
     {"check_config": CallCount.ONE, "prepare_certs": CallCount.ZERO}),

    ({
        "address": "localhost",
        "port": 9000,
        "tls_config": {
            "server_cert_path": PATH_VALID,
            "client_key_path": PATH_VALID
        }
    },
     ValueError, 'none or both client_key_path and client_cert_path are required in tls_config',
     {"check_config": CallCount.ONE, "prepare_certs": CallCount.ZERO}),

    ({
        "address": "localhost",
        "port": 9000,
        "tls_config": {
            "server_cert_path": PATH_VALID,
            "client_key_path": PATH_VALID,
            "client_cert_path": PATH_VALID,
            "invalid_key_name": PATH_VALID
        }
    },
     ValueError,  'invalid_key_name is not valid tls_config key',
     {"check_config": CallCount.ONE, "prepare_certs": CallCount.ZERO}),

    ({
        "address": "localhost",
        "port": 9000,
        "tls_config": {
            "server_cert_path": PATH_VALID,
            "client_key_path": PATH_VALID,
            "client_cert_path": 123,
        }
    },
     TypeError,  'client_cert_path type should be string but is type int',
     {"check_config": CallCount.ONE, "prepare_certs": CallCount.ZERO}),

    ({
        "address": "localhost",
        "port": 9000,
        "tls_config": {
            "server_cert_path": PATH_VALID,
            "client_key_path": "invalid_path",
            "client_cert_path": PATH_VALID,
        }
    },
     ValueError,  'invalid_path is not valid path to file',
     {"check_config": CallCount.ONE, "prepare_certs": CallCount.ZERO}),

]

# (config_dict,
# expected_exception, expected_message,
# method_call_count_dict= {"method_name": CallCount.NumberOfCalls})
BUILD_INVALID_CERTS = [
    ({
        "address": "localhost",
        "port": 9000,
        "tls_config": {
            "server_cert_path": PATH_VALID,
            "client_key_path": "path_to_invalid_private_key",
            "client_cert_path": PATH_VALID,
        }
    },
     ValueError,  'path_to_invalid_private_key file is not valid private key',
     {"check_config": CallCount.ONE, "prepare_certs": CallCount.ONE}),

    ({
        "address": "localhost",
        "port": 9000,
        "tls_config": {
            "server_cert_path": "path_to_invalid_server_certificate",
            "client_key_path": PATH_VALID,
            "client_cert_path": PATH_VALID,
        }
    },
     ValueError,  'path_to_invalid_server_certificate is not valid certificate',
     {"check_config": CallCount.ONE, "prepare_certs": CallCount.ONE}),

    ({
        "address": "localhost",
        "port": 9000,
        "tls_config": {
            "server_cert_path": PATH_VALID,
            "client_key_path": PATH_VALID,
            "client_cert_path": "path_to_invalid_client_certificate",
        }
    },
     ValueError,  'path_to_invalid_client_certificate is not valid certificate',
     {"check_config": CallCount.ONE, "prepare_certs": CallCount.ONE}),
]
