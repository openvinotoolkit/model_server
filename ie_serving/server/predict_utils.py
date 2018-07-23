from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import tensor_shape
from ie_serving.tensorflow_serving_api import predict_pb2
# import tensorflow.contrib.util as tf_contrib_util
from tensorflow.python.framework import dtypes as dtypes
from tensorflow.python.framework import tensor_util as tensor_util


def check_if_model_name_and_version_is_valid(model_name, version,
                                             available_models):
    if model_name in list(available_models.keys()):
        if version == 0:
            return False
        else:
            return True
    return False


def get_version_model(model_name, requested_version, available_models):
    version = 0
    requested_version = int(requested_version)
    if model_name in available_models:
        if requested_version == 0:
            version = available_models[model_name].default_version
        elif requested_version in available_models[model_name].versions:
            version = requested_version
    return version


def prepare_output_as_list(inference_output, model_available_outputs):
    response = predict_pb2.PredictResponse()
    for output in model_available_outputs:
        dtype = dtypes.as_dtype(inference_output[output].dtype)
        output_tensor = tensor_pb2.TensorProto(
            dtype=dtype.as_datatype_enum,
            tensor_shape=tensor_shape.as_shape(
                inference_output[output].shape).as_proto())
        result = inference_output[output].flatten()
        tensor_util._NP_TO_APPEND_FN[dtype.as_numpy_dtype](output_tensor,
                                                           result)
        response.outputs[output].CopyFrom(output_tensor)
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
