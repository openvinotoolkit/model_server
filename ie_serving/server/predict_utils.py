#
# Copyright (c) 2018 Intel Corporation
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
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import tensor_shape
from ie_serving.tensorflow_serving_api import predict_pb2
from tensorflow.python.framework import dtypes as dtypes
from tensorflow.python.framework import tensor_util as tensor_util
import tensorflow.contrib.util as tf_contrib_util
# import tensorflow.contrib.util as tf_contrib_util
from ie_serving.server.constants import INVALID_INPUT_KEY, INVALID_SHAPE
from ie_serving.logger import get_logger

logger = get_logger(__name__)


def prepare_input_data(models, model_name, version, data):
    model_inputs_in_input_request = list(dict(data).keys())
    input_keys = models[model_name].engines[version].input_key_names
    inference_input = {}

    for requested_input_blob in model_inputs_in_input_request:
        if requested_input_blob not in input_keys:
            code = StatusCode.INVALID_ARGUMENT
            message = INVALID_INPUT_KEY % (model_inputs_in_input_request,
                                           input_keys)
            logger.debug("PREDICT error: {}".format(message))
            return True, message,None, code

        tensor_name = models[model_name].engines[version]. \
            model_keys['inputs'][requested_input_blob]
        try:
            tensor_input = tf_contrib_util. \
                make_ndarray(data[requested_input_blob])
        except Exception as e:
            code = StatusCode.INVALID_ARGUMENT
            message = str(e)
            logger.debug("PREDICT prepare_input_data make_ndarray error: {}"
                         .format(message))
            return True, message, code

        shape_required_in_model = models[model_name].engines[version] \
            .input_tensors[tensor_name].shape
        # check requested shape and model shape
        if shape_required_in_model != list(tensor_input.shape):
            code = StatusCode.INVALID_ARGUMENT
            message = INVALID_SHAPE.format(list(tensor_input.shape),
                                           shape_required_in_model)
            logger.debug("PREDICT error: {}".format(message))
            return True, message, None, code
        inference_input[tensor_name] = tensor_input
        batch_size = shape_required_in_model[0]
    return False, inference_input, batch_size, None


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
