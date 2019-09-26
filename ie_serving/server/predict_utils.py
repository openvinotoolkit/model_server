#
# Copyright (c) 2018-2019 Intel Corporation
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
import numpy as np
from grpc import StatusCode
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import tensor_shape
from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import dtypes as dtypes
from tensorflow.python.framework import tensor_util as tensor_util
import tensorflow.contrib.util as tf_contrib_util
# import tensorflow.contrib.util as tf_contrib_util
from ie_serving.models.shape_management.utils import BatchingMode, ShapeMode
from ie_serving.server.constants import \
    INVALID_INPUT_KEY, INVALID_SHAPE, INVALID_BATCHSIZE, GRPC, REST
from ie_serving.logger import get_logger

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
                tensor_input = tf_contrib_util. \
                    make_ndarray(data[requested_input_blob])
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


def prepare_output_as_list(inference_output, model_available_outputs):
    response = predict_pb2.PredictResponse()
    for key, value in model_available_outputs.items():
        if value in inference_output:
            dtype = dtypes.as_dtype(inference_output[value].dtype)
            output_tensor = tensor_pb2.TensorProto(
                dtype=dtype.as_datatype_enum,
                tensor_shape=tensor_shape.as_shape(
                    inference_output[value].shape).as_proto())
            result = inference_output[value].flatten()
            tensor_util._NP_TO_APPEND_FN[dtype.as_numpy_dtype](output_tensor,
                                                               result)
            response.outputs[key].CopyFrom(output_tensor)
    return response


'''
The function is not used.
Probably preparing the output would be faster,
but you need a change of grpc clients.

def prepare_output_with_tf(inference_output, model_available_outputs):
    response = predict_pb2.PredictResponse()

    for output in model_available_outputs:
        response.outputs[output].CopyFrom(
            tf_contrib_util.make_tensor_proto(inference_output[output],
                                              shape=inference_output[output].
                                              shape,
                                              dtype=dtypes.as_dtype(
                                                  inference_output
                                                  [output].dtype).
                                              as_datatype_enum))
    return response
'''
