from ie_serving.tensorflow_serving_api import prediction_service_pb2
from ie_serving.tensorflow_serving_api import predict_pb2
import tensorflow.contrib.util as tf_contrib_util
import grpc
from ie_serving.server.predict_utils import \
    check_if_model_name_and_version_is_valid, prepare_output_as_list, \
    get_version_model


class PredictionServiceServicer(prediction_service_pb2.
                                PredictionServiceServicer):

    def __init__(self, models):
        self.models = models

    def Predict(self, request, context):
        """
        Predict -- provides access to loaded TensorFlow model.
        """
        # check if model with was requested
        # is available on server with proper version
        model_name = request.model_spec.name
        version = get_version_model(model_name=model_name,
                                    requested_version=request.model_spec.
                                    version.value,
                                    available_models=self.models)
        valid_model_spec = check_if_model_name_and_version_is_valid(
            model_name=model_name, version=version,
            available_models=self.models)

        if not valid_model_spec:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Servable not found for request: '
                                'Specific({}, {})'.format(model_name, version))
            return predict_pb2.PredictResponse()
        model_inputs_in_input_request = list(dict(request.inputs).keys())
        input_blob = self.models[model_name].engines[version].input_blob
        # check if name of requested blob is equal model blob
        if input_blob not in model_inputs_in_input_request:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('input tensor alias not found in signature: '
                                '%s. Inputs expected to be in the set {%s}.'
                                % (model_inputs_in_input_request, input_blob))
            return predict_pb2.PredictResponse()

        inference_input = tf_contrib_util.make_ndarray(request.
                                                       inputs[input_blob])
        shape_required_in_model = self.models[model_name].engines[version]\
            .inputs[input_blob]
        # check requested shape and model shape
        if shape_required_in_model != list(inference_input.shape):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('The input data is incorrect. '
                                'Obtained shape {}, required shape {}'.
                                format(list(inference_input.shape),
                                       shape_required_in_model))
            return predict_pb2.PredictResponse()
        inference_input = {input_blob: inference_input}
        inference_output = self.models[model_name].engines[version]\
            .infer(inference_input)
        response = prepare_output_as_list(inference_output=inference_output,
                                          model_available_outputs=self.models
                                          [model_name].engines[version].
                                          outputs)
        response.model_spec.name = model_name
        response.model_spec.version.value = version
        response.model_spec.signature_name = "serving_default"
        return response
