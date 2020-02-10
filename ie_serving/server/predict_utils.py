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

logger = get_logger(__name__)

statusCodes = {
    'invalid_arg': {GRPC: StatusCode.INVALID_ARGUMENT,
                    REST: falcon.HTTP_BAD_REQUEST},
}


def prepare_input_data(target_engine, data, service_type):
    # returns:
    # inference_input, None on success
    # None, error_message on error
    model_inputs_in_input_request = list(dict(data).keys())
    input_keys = target_engine.input_key_names
    inference_input = {}

    for requested_input_blob in model_inputs_in_input_request:
        if requested_input_blob not in input_keys:
            message = INVALID_INPUT_KEY % (model_inputs_in_input_request,
                                           input_keys)
            logger.debug("PREDICT error: {}".format(message))
            return None, message

        tensor_name = target_engine.model_keys['inputs'][requested_input_blob]
        if service_type == GRPC:
            try:
                tensor_input = make_ndarray(data[requested_input_blob])
            except Exception as e:
                message = str(e)
                logger.debug("PREDICT prepare_input_data make_ndarray error: "
                             "{}".format(message))
                return None, message
        else:
            tensor_input = np.asarray(data[requested_input_blob])
        # Validate shape if shape not in auto mode
        if target_engine.shape_info.mode != ShapeMode.AUTO:
            shape_required_in_model = target_engine.net.inputs[
                tensor_name].shape

            # For reshapable models check all dimensions,
            # for non-reshapable, check all starting from the second (omit
            # batch size)
            if target_engine.shape_info.mode == ShapeMode.DISABLED:
                starting_dim = 1
            else:
                starting_dim = 0

            # check requested shape and model shape
            if shape_required_in_model[starting_dim:] != list(
                    tensor_input.shape)[starting_dim:]:
                message = INVALID_SHAPE.format(list(tensor_input.shape),
                                               shape_required_in_model)
                logger.debug("PREDICT error: {}".format(message))
                return None, message

            # check if input batch size match the model only if not auto mode
            if target_engine.batching_info.mode != \
                BatchingMode.AUTO and shape_required_in_model[0] != \
                    tensor_input.shape[0]:
                message = INVALID_BATCHSIZE.format(
                    tensor_input.shape[0],
                    target_engine.batching_info.batch_size)
                logger.debug("PREDICT error,Invalid batchsize:{}".format(
                    message))
                return None, message

        inference_input[tensor_name] = tensor_input
    return inference_input, None


'''
function _prepare_output_as_AppendArrayToTensorProto returns inference
results in a form of flattened list of array elements. It is serialized
using tensor_util._NP_TO_APPEND_FN function which employs module
fast_tensor_util and functions Append<dtype>ArrayToTensorProto.
Despite the module name, it is slower from make_tensor_proto.
'''


def _prepare_output_as_AppendArrayToTensorProto(
        inference_output,
        model_available_outputs):
    response = predict_pb2.PredictResponse()
    for response_output_name, model_output_name in \
            model_available_outputs.items():
        if model_output_name in inference_output:
            dtype = dtypes.as_dtype(inference_output[model_output_name].dtype)
            output_tensor = tensor_pb2.TensorProto(
                dtype=dtype.as_datatype_enum,
                tensor_shape=tensor_shape.as_shape(
                    inference_output[model_output_name].shape).as_proto())
            result = inference_output[model_output_name].flatten()
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


def _prepare_output_with_make_tensor_proto(
        inference_output,
        model_available_outputs):
    response = predict_pb2.PredictResponse()
    for response_output_name in model_available_outputs:
        model_output_name = model_available_outputs[response_output_name]
        response.outputs[response_output_name].CopyFrom(
            make_tensor_proto(inference_output[model_output_name]))
    return response


# Serialization method selection
prepare_output = None
if GLOBAL_CONFIG['serialization_function'] == \
        '_prepare_output_as_AppendArrayToTensorProto':
    prepare_output = _prepare_output_as_AppendArrayToTensorProto
else:
    prepare_output = _prepare_output_with_make_tensor_proto
