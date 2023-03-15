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
from ovmsclient.tfs_compat.base.errors import ModelNotFoundError, InvalidInputError

from config import CallCount, PATH_VALID # noqa

from ovmsclient.tfs_compat.protos.tensorflow.core.framework.tensor_pb2 import TensorProto
from ovmsclient.tfs_compat.protos.tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto # noqa
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.get_model_status_pb2 import ModelVersionStatus # noqa
from ovmsclient.tfs_compat.protos.tensorflow.core.framework.types_pb2 import DataType
from ovmsclient.tfs_compat.protos.tensorflow.core.protobuf.error_codes_pb2 import Code as ErrorCode
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusRequest # noqa
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.get_model_metadata_pb2 import GetModelMetadataRequest # noqa
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.predict_pb2 import PredictRequest

from ovmsclient.tfs_compat.grpc.requests import (GrpcModelMetadataRequest, GrpcModelStatusRequest,
                                                 GrpcPredictRequest)

# responses_dict = {
#    model_version: { expected_status }
# }
MODEL_STATUS_RESPONSE_VALID = [
    {
        1: {
            "state": ModelVersionStatus.State.AVAILABLE,
            "error_code": ErrorCode.OK,
            "error_message": "OK"
        }
    },
    {
        2: {
            "state": ModelVersionStatus.State.END,
            "error_code": ErrorCode.OK,
            "error_message": "OK"
        },
        3: {
            "state": ModelVersionStatus.State.AVAILABLE,
            "error_code": ErrorCode.OK,
            "error_message": ""
        }
    },
    {
        1: {
            "state": ModelVersionStatus.State.START,
            "error_code": ErrorCode.OK,
            "error_message": ""
        },
        2: {
            "state": ModelVersionStatus.State.LOADING,
            "error_code": ErrorCode.UNKNOWN,
            "error_message": "Could not load CNN"
        },
        3: {
            "state": ModelVersionStatus.State.UNLOADING,
            "error_code": ErrorCode.OK,
            "error_message": ""
        }
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
    }, 'model_name', 0, TypeError, "inputs keys type should be str, but found int"),
    ({
        "input1": [[1.0, 2.0], [1.0, 2.0, 3.0]]
    }, 'model_name', 0, ValueError,
     ("setting an array element with a sequence. "
      "The requested array has an inhomogeneous shape after 1 dimensions. "
      "The detected shape was (2,) + inhomogeneous part.")),
    ({
        "input1": [[(1, 2, 3)], [(1, 2)], [(1, 2, 3)]]
    }, 'model_name', 0, ValueError,
     ("setting an array element with a sequence. "
      "The requested array has an inhomogeneous shape after 2 dimensions. "
      "The detected shape was (3, 1) + inhomogeneous part.")),
    ({
        "input1": float128(2.5)
    }, 'model_name', 0, TypeError, "provided values type is not valid"),
    ({
        "input1": (1, 2, 3)
    }, 'model_name', 0, TypeError,
     "values type should be (list, np.ndarray, scalar), but is tuple"),
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
        "input3": bytes([1, 2, 3]),
        "input4": [[bytes([1, 2, 3]), bytes([1, 2, 3])], [bytes([1, 2, 3]), bytes([1, 2, 3])]],
        "input5": [["list", "of", "strings"]],
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
        },
        "input4": {
            "field": "string_val",
            "shape": TensorShapeProto(dim=[TensorShapeProto.Dim(size=2),
                                           TensorShapeProto.Dim(size=2)]),
            "dtype": DataType.DT_STRING,
            'value': [bytes([1, 2, 3]), bytes([1, 2, 3]), bytes([1, 2, 3]), bytes([1, 2, 3])]
        },
        "input5": {
            "field": "string_val",
            "shape": TensorShapeProto(dim=[TensorShapeProto.Dim(size=1),
                                           TensorShapeProto.Dim(size=3)]),
            "dtype": DataType.DT_STRING,
            'value': [b'list', b'of', b'strings']
        },
    }, 'model_name', 0),

    ({

    }, {

    }, 'model_name', 0)
]

# (response_outputs_dict, model_name, model_version, expected_outputs)
PREDICT_RESPONSE_VALID = [
    ({
        "1463": TensorProto(dtype=DataType.DT_INT8,
                            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]),
                            tensor_content=array([1, 2, 3], dtype=int8).tobytes()),
    }, "model_name", 0, array([1, 2, 3], dtype=int8)
    ),

    ({
        "1463": TensorProto(dtype=DataType.DT_INT32,
                            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=2),
                                                          TensorShapeProto.Dim(size=3)]),
                            tensor_content=array([1, 2, 3, 4, 5, 6], dtype=int32).tobytes()),
        "2": TensorProto(dtype=DataType.DT_DOUBLE,
                         tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
                         double_val=array([12.0], dtype=float64)),
    }, "model_name", 0, {
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
        "3": TensorProto(dtype=DataType.DT_STRING,
                         tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1),
                                                            TensorShapeProto.Dim(size=3)]),
                         string_val=[b'list', b'of', b'strings']),
        "4": TensorProto(dtype=DataType.DT_STRING,
                         tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=2),
                                                            TensorShapeProto.Dim(size=2)]),
                         string_val=[bytes([1, 2, 3]), bytes([1, 2, 3]),
                                     bytes([1, 2, 3]), bytes([1, 2, 3])]),
    }, "model_name", 0, {
        "1463": [bytes([1, 2, 3]), bytes([4, 5])],
        "2": [bytes([1, 2, 3])],
        "3": [[b'list', b'of', b'strings']],
        "4": [[bytes([1, 2, 3]), bytes([1, 2, 3])], [bytes([1, 2, 3]), bytes([1, 2, 3])]]
    }),
]

# (response_outputs_dict, model_name, model_version, expected_exception, expected_message)
PREDICT_RESPONSE_TENSOR_TYPE_INVALID = [
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

# (grpc_error_status_code, grpc_error_details, raised_error_type, raised_error_message)
COMMON_INVALID_GRPC = [
    (StatusCode.UNAVAILABLE, "failed to connect to all adresses",
     ConnectionError, "Error occurred during handling the request: "
                      "failed to connect to all adresses"),
    (StatusCode.UNAVAILABLE, "Empty update",
     ConnectionError, "Error occurred during handling the request: Empty update"),
    (StatusCode.DEADLINE_EXCEEDED, "Deadline Exceeded",
     TimeoutError, "Error occurred during handling the request: "
                   "Request handling exceeded timeout"),
    (StatusCode.NOT_FOUND, "Model with requested version is not found",
     ModelNotFoundError, "Error occurred during handling the request: "
                         "Model with requested version is not found"),
    (StatusCode.NOT_FOUND, "Model with requested name is not found",
     ModelNotFoundError, "Error occurred during handling the request: "
                         "Model with requested name is not found"),
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

# (grpc_error_status_code, grpc_error_details, raised_error_type, raised_error_message)
PREDICT_INVALID_GRPC = COMMON_INVALID_GRPC + [
    (StatusCode.INVALID_ARGUMENT, "Invalid input precision - Expected: FP32; Actual: I64",
     InvalidInputError, "Error occurred during handling the request: "
                        "Invalid input precision - Expected: FP32; Actual: I64"),
    (StatusCode.INVALID_ARGUMENT, "Invalid number of inputs - Expected: 1; Actual: 0",
     InvalidInputError, "Error occurred during handling the request: "
                        "Invalid number of inputs - Expected: 1; Actual: 0"),
    (StatusCode.INVALID_ARGUMENT, "Missing input with specific name - Required input: 0",
     InvalidInputError, "Error occurred during handling the request: "
                        "Missing input with specific name - Required input: 0"),
    (StatusCode.INVALID_ARGUMENT, "Invalid number of shape dimensions - "
                                  "Expected: (1,3,224,224); Actual: (3)",
     InvalidInputError, "Error occurred during handling the request: "
                        "Invalid number of shape dimensions - Expected: (1,3,224,224); "
                        "Actual: (3)"),
]

# (config_dict,
# method_call_count_dict= {"method_name": CallCount.NumberOfCalls})
BUILD_VALID = [
    (
        {
            "url": "localhost:9000"
        },
        {
            "_check_url": CallCount.ONE,
            "_check_tls_config": CallCount.ZERO,
            "_prepare_certs": CallCount.ZERO
        }
    ),
    (
        {
            "url": "19.117.63.126:1"
        },
        {
            "_check_url": CallCount.ONE,
            "_check_tls_config": CallCount.ZERO,
            "_prepare_certs": CallCount.ZERO
        }
    ),
    (
        {
            "url": f"cluster.cloud.iotg.intel.com:{2**16-1}"
        },
        {
            "_check_url": CallCount.ONE,
            "_check_tls_config": CallCount.ZERO,
            "_prepare_certs": CallCount.ZERO
        }
    ),
    (
        {
            "url": "modelserver.default.svc.cluster.local:9000000000000121321312312",
            "tls_config": {
                "server_cert_path": "valid_path"
            }
        },
        {
            "_check_url": CallCount.ONE,
            "_check_tls_config": CallCount.ONE,
            "_prepare_certs": CallCount.ONE
        }
    ),
    (
        {
            "url": "modelserver123:9000",
            "tls_config": {
                "client_key_path": PATH_VALID,
                "client_cert_path": PATH_VALID,
                "server_cert_path": PATH_VALID
            }
        },
        {
            "_check_url": CallCount.ONE,
            "_check_tls_config": CallCount.ONE,
            "_prepare_certs": CallCount.ONE
        }
    )
]

# (config_dict,
# method_call_dict= {"method_name": (CallCount.NumberOfCalls, error_raised)},
# expected_exception, expected_message)
BUILD_INVALID_CONFIG = [
    (
        {
            "url": None
        },
        {
            "_check_url": (CallCount.ONE, TypeError("url must be a string")),
            "_check_tls_config": (CallCount.ZERO, None),
            "_prepare_certs": (CallCount.ZERO, None)
        },
        TypeError, "url must be a string"
    ),

    (
        {
            "url": 123
        },
        {
            "_check_url": (CallCount.ONE, TypeError("url must be a string")),
            "_check_tls_config": (CallCount.ZERO, None),
            "_prepare_certs": (CallCount.ZERO, None)
        },
        TypeError, "url must be a string"
    ),

    (
        {
            "url": ["address:9000"],
        },
        {
            "_check_url": (CallCount.ONE, TypeError("url must be a string")),
            "_check_tls_config": (CallCount.ZERO, None),
            "_prepare_certs": (CallCount.ZERO, None)
        },
        TypeError, "url must be a string"
    ),

    (
        {
            "url": {"address": "127.0.0.1", "port": 9000}
        },
        {
            "_check_url": (CallCount.ONE, TypeError("url must be a string")),
            "_check_tls_config": (CallCount.ZERO, None),
            "_prepare_certs": (CallCount.ZERO, None)
        },
        TypeError, "url must be a string"
    ),

    (
        {
            "url": [123, 546]
        },
        {
            "_check_url": (CallCount.ONE, TypeError("url must be a string")),
            "_check_tls_config": (CallCount.ZERO, None),
            "_prepare_certs": (CallCount.ZERO, None)
        },
        TypeError, "url must be a string"
    ),

    (
        {
            "url": "localhost:9000",
            "tls_config": 123
        },
        {
            "_check_url": (CallCount.ONE, None),
            "_check_tls_config": (CallCount.ONE, TypeError("tls_config should be of type dict")),
            "_prepare_certs": (CallCount.ZERO, None)
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
            "_prepare_certs": (CallCount.ZERO, None)
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
            "_check_tls_config": (CallCount.ONE, ValueError("none or both client_key_path "
                                                            "and client_cert_path are required "
                                                            "in tls_config")),
            "_prepare_certs": (CallCount.ZERO, None)
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
            "_check_tls_config": (CallCount.ONE, ValueError("invalid_key_name is "
                                                            "not valid tls_config key")),
            "_prepare_certs": (CallCount.ZERO, None)
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
            "_prepare_certs": (CallCount.ZERO, None)
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
            "_prepare_certs": (CallCount.ZERO, None)
        },
        ValueError,  "invalid_path is not valid path to file"
    ),

]

# (config_dict,
# method_call_dict= {"method_name": (CallCount.NumberOfCalls, error_raised)},
# expected_exception, expected_message)
BUILD_INVALID_CERTS = [
    (
        {
            "url": "localhost:9000",
            "tls_config": {
                "server_cert_path": PATH_VALID,
                "client_key_path": "path_to_invalid_private_key",
                "client_cert_path": PATH_VALID,
            }
        },
        {
            "_check_url": (CallCount.ONE, None),
            "_check_tls_config": (CallCount.ONE, None),
            "_prepare_certs": (CallCount.ONE, ValueError("path_to_invalid_private_key file "
                                                         "is not valid private key"))
        },
        ValueError, "path_to_invalid_private_key file is not valid private key"
    ),

    (
        {
            "url": "localhost:9000",
            "tls_config": {
                "server_cert_path": "path_to_invalid_server_certificate",
                "client_key_path": PATH_VALID,
                "client_cert_path": PATH_VALID,
            }
        },
        {
            "_check_url": (CallCount.ONE, None),
            "_check_tls_config": (CallCount.ONE, None),
            "_prepare_certs": (CallCount.ONE, ValueError("path_to_invalid_server_certificate "
                                                         "is not valid certificate"))
        },
        ValueError, "path_to_invalid_server_certificate is not valid certificate"
    ),

    (
        {
            "url": "localhost:9000",
            "tls_config": {
                "server_cert_path": PATH_VALID,
                "client_key_path": PATH_VALID,
                "client_cert_path": "path_to_invalid_client_certificate",
            }
        },
        {
            "_check_url": (CallCount.ONE, None),
            "_check_tls_config": (CallCount.ONE, None),
            "_prepare_certs": (CallCount.ONE, ValueError("path_to_invalid_client_certificate "
                                                         "is not valid certificate"))
        },
        ValueError,  "path_to_invalid_client_certificate is not valid certificate"
    ),
]
