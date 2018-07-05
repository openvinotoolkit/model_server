from ie_serving.tensorflow_serving_api import prediction_service_pb2, predict_pb2
import tensorflow as tf
from tensorflow.core.framework import tensor_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import tensor_shape
import grpc
from ie_serving.server.parsers import check_if_model_name_and_version_is_valid


class PredictionServiceServicer(prediction_service_pb2.PredictionServiceServicer):

    def __init__(self, models):
        self.models = models

    def Predict(self, request, context):
        """
        Predict -- provides access to loaded TensorFlow model.
        """
        valid_model_spec, model_name, version = check_if_model_name_and_version_is_valid(model_spec=request.model_spec,
                                                                                         available_models=self.models)

        model_inputs_in_input_request = list(dict(request.inputs).keys())
        if valid_model_spec:
            input_blob = self.models[model_name].engines[version].input_blob
            if input_blob in model_inputs_in_input_request:
                inference_input = tf.contrib.util.make_ndarray(request.inputs[input_blob])
                if self.models[model_name].engines[version].inputs[input_blob] == list(inference_input.shape):
                    inference_output = self.models[model_name].engines[version].infer(inference_input)
                    response = predict_pb2.PredictResponse()
                    for output in self.models[model_name].engines[version].outputs:
                        output_tensor = tensorflow_dot_core_dot_framework_dot_tensor__pb2.TensorProto(
                            dtype=types_pb2.DT_FLOAT,
                            tensor_shape=tensor_shape.as_shape(inference_output[output].shape).as_proto())
                        for result in inference_output[output]:
                            output_tensor.float_val.extend(result)
                        response.outputs[output].CopyFrom(output_tensor)
                        response.model_spec.name = model_name
                        response.model_spec.version.value = version
                        response.model_spec.signature_name = "serving_default"
                        return response
                else:
                    print("test")

            else:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details('input tensor alias not found in signature: %s. '
                                    'Inputs expected to be in the set {%s}.' % (model_inputs_in_input_request, input_blob))
            '''
            response.outputs['out'].CopyFrom(
            tf.contrib.util.make_tensor_proto(test['resnet_v1_50/predictions/Reshape_1'],
                                          shape=test['resnet_v1_50/predictions/Reshape_1'].shape,
                                          dtype=types_pb2.DT_FLOAT))
            '''

        else:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Servable not found for request: Specific({}, {})'.format(model_name, version))
        raise NotImplementedError('Method not implemented!')
