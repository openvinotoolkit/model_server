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
from ie_serving.tensorflow_serving_api import predict_pb2
from tensorflow.python.framework import dtypes as dtypes
from tensorflow.python.framework import tensor_util as tensor_util
import tensorflow.contrib.util as tf_contrib_util
# import tensorflow.contrib.util as tf_contrib_util
from ie_serving.server.constants import \
    INVALID_INPUT_KEY, INVALID_SHAPE, INVALID_BATCHSIZE
from ie_serving.logger import get_logger

logger = get_logger(__name__)

statusCodes = {
    'invalid_arg': {'grpc': StatusCode.INVALID_ARGUMENT,
                    'rest': falcon.HTTP_BAD_REQUEST},
}


def extract_list(dictionary):
    for value in dictionary.values():
        return value


def is_list_of_dicts(list_of_objects):
    if type(list_of_objects[0]) is dict:
        return True
    else:
        return False


def row_to_column(list_of_dicts):
    output_dict = dict()
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            if key not in output_dict.keys():
                output_dict[key] = []
            output_dict[key].append(value)
    return output_dict


def column_to_row(dict_of_lists):
    output_list = []
    for values in zip(*dict_of_lists.values()):
        dictionary = dict()
        for (key, value) in zip(dict_of_lists.keys(), values):
            dictionary[key] = value
        output_list.append(dictionary)
    return output_list


def prepare_json_response(request_body, inference_output):
    if "instances" in request_body.keys():
        if is_list_of_dicts(request_body['instances']):
            if len(request_body['instances'][0].keys()) > 1:
                response = {'predictions': column_to_row(inference_output)}
            else:
                response = {'predictions': extract_list(inference_output)}
        else:
            response = {'predictions': extract_list(inference_output)}
    else:
        if type(request_body['inputs']) is dict:
            if len(request_body['inputs'].keys()) > 1:
                response = {'outputs': inference_output}
            else:
                response = {'outputs': extract_list(inference_output)}
        else:
            response = {'outputs': extract_list(inference_output)}
    return response


def prepare_input_data(models, model_name, version, data, rest):
    if type(data) is list:
        model_inputs_in_input_request = models[model_name].engines[
            version].input_key_names
        data = {model_inputs_in_input_request[0]: data}
    else:
        model_inputs_in_input_request = list(dict(data).keys())
    input_keys = models[model_name].engines[version].input_key_names
    inference_input = {}
    request_type = 'grpc' if not rest else 'rest'

    for requested_input_blob in model_inputs_in_input_request:
        if requested_input_blob not in input_keys:
            code = statusCodes['invalid_arg'][request_type]
            message = INVALID_INPUT_KEY % (model_inputs_in_input_request,
                                           input_keys)
            logger.debug("PREDICT error: {}".format(message))
            return True, message, None, code

        tensor_name = models[model_name].engines[version]. \
            model_keys['inputs'][requested_input_blob]
        if not rest:
            try:
                tensor_input = tf_contrib_util. \
                    make_ndarray(data[requested_input_blob])
            except Exception as e:
                code = statusCodes['invalid_arg'][request_type]
                message = str(e)
                logger.debug("PREDICT prepare_input_data make_ndarray error: "
                             "{}".format(message))
                return True, message, None, code
        else:
            tensor_input = np.asarray(data[requested_input_blob])

        shape_required_in_model = models[model_name].engines[version] \
            .input_tensors[tensor_name].shape
        # check if input batch size match the model only if not auto mode
        if models[model_name].engines[version].batch_size != 0 \
                and shape_required_in_model[0] != tensor_input.shape[0]:
            code = statusCodes['invalid_arg'][request_type]
            message = INVALID_BATCHSIZE.format(
                tensor_input.shape[0],
                models[model_name].engines[version].batch_size)
            logger.debug("PREDICT error,Invalid batchsize:{}".format(message))
            return True, message, None, code

        # check requested shape and model shape
        if shape_required_in_model[1:] != list(tensor_input.shape)[1:]:
            code = statusCodes['invalid_arg'][request_type]
            message = INVALID_SHAPE.format(list(tensor_input.shape),
                                           shape_required_in_model)
            logger.debug("PREDICT error: {}".format(message))
            return True, message, None, code
        inference_input[tensor_name] = tensor_input
        batch_size = list(tensor_input.shape)[0]
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
