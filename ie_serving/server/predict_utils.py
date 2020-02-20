#
# Copyright (c) 2018-2020 Intel Corporation
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

import falcon
from grpc import StatusCode
import numpy as np
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import tensor_shape
from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import dtypes as dtypes
from tensorflow.python.framework import tensor_util as tensor_util
from ie_serving.config import GLOBAL_CONFIG
from ie_serving.models.shape_management.utils import BatchingMode, ShapeMode
from ie_serving.server.constants import \
    INVALID_INPUT_KEY, INVALID_SHAPE, INVALID_BATCHSIZE, GRPC, REST
from ie_serving.logger import get_logger
from tensorflow import __version__ as tf_version
if tf_version.split(".")[0] == "2":
    from tensorflow import make_ndarray, make_tensor_proto
else:  # TF version 1.x
    from tensorflow.contrib.util import make_ndarray, make_tensor_proto

from multiprocessing import shared_memory
from ie_serving.messaging.endpoint_requests_pb2 import EndpointRequest, \
    PredictRequest
from ie_serving.messaging.data_attributes_pb2 import DataAttributes

logger = get_logger(__name__)

statusCodes = {
    'invalid_arg': {GRPC: StatusCode.INVALID_ARGUMENT,
                    REST: falcon.HTTP_BAD_REQUEST},
}


def prepare_ipc_predict_request(data_type, data, return_socket_name):
    # TODO: handling various data types
    ipc_endpoint_request = EndpointRequest()
    ipc_predict_request = PredictRequest()
    allocated_shm_names = []
    ipc_inputs = []

    inputs = dict(data)
    for input_name in list(inputs.keys()):
        single_input = make_ndarray(inputs[input_name])

        input_shm_name = shared_memory.SharedMemory(
            create=True, size=single_input.nbytes)
        shm_array = np.ndarray(single_input.shape, dtype=single_input.dtype,
                               buffer=single_input.buf)
        shm_array[:] = single_input[:]

        ipc_numpy_attributes = DataAttributes.NumpyAttributes()
        ipc_numpy_attributes.shape.extend(list(shm_array.shape))
        ipc_numpy_attributes.data_type = shm_array.dtype.name

        ipc_data_attributes = DataAttributes()
        ipc_data_attributes.numpy_attributes.CopyFrom(ipc_numpy_attributes)

        ipc_input_data = PredictRequest.Data()
        ipc_input_data.attributes.CopyFrom(ipc_data_attributes)
        ipc_input_data.input_name = input_name
        ipc_input_data.shm_name = input_shm_name
        ipc_inputs.append(ipc_input_data)

        allocated_shm_names.append(input_shm_name)

    ipc_predict_request.return_socket_name = return_socket_name
    ipc_predict_request.inputs.extend(ipc_inputs)
    ipc_endpoint_request.predict_request.CopyFrom(ipc_predict_request)
    return ipc_endpoint_request, allocated_shm_names

'''
function _prepare_output_as_AppendArrayToTensorProto returns inference
results in a form of flattened list of array elements. It is serialized
using tensor_util._NP_TO_APPEND_FN function which employs module
fast_tensor_util and functions Append<dtype>ArrayToTensorProto.
Despite the module name, it is slower from make_tensor_proto.
'''


def _prepare_output_as_AppendArrayToTensorProto(
        inference_output):
    response = predict_pb2.PredictResponse()
    for response_output_name in inference_output:
        dtype = dtypes.as_dtype(inference_output[response_output_name].dtype)
        output_tensor = tensor_pb2.TensorProto(
            dtype=dtype.as_datatype_enum,
            tensor_shape=tensor_shape.as_shape(
                inference_output[response_output_name].shape).as_proto())
        result = inference_output[response_output_name].flatten()
        tensor_util._NP_TO_APPEND_FN[dtype.as_numpy_dtype](output_tensor,
                                                               result)
        response.outputs[response_output_name].CopyFrom(output_tensor)
    return response


'''
The function prepare_output_with_make_tensor_proto implements faster
serialization mechanism. For most of the models it will return
data in string format converted via numpy.toString calls.
On the client side the inference response can be deserialized using
Tensorflow make_ndarray function.
'''


def _prepare_output_with_make_tensor_proto(inference_output):
    response = predict_pb2.PredictResponse()
    for response_output_name in inference_output:
        response.outputs[response_output_name].CopyFrom(
            make_tensor_proto(inference_output[response_output_name]))
    return response


# Serialization method selection
prepare_output = None
if GLOBAL_CONFIG['serialization_function'] == \
        '_prepare_output_as_AppendArrayToTensorProto':
    prepare_output = _prepare_output_as_AppendArrayToTensorProto
else:
    prepare_output = _prepare_output_with_make_tensor_proto
