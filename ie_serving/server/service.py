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

from ie_serving.tensorflow_serving_api import prediction_service_pb2
from ie_serving.tensorflow_serving_api import predict_pb2
from ie_serving.tensorflow_serving_api import get_model_metadata_pb2

from ie_serving.server.service_utils import \
    check_availability_of_requested_model
from ie_serving.server.get_model_metadata_utils import \
    prepare_get_metadata_output
from ie_serving.server.predict_utils import prepare_output_as_list, \
    prepare_input_data, StatusCode
from ie_serving.server.constants import WRONG_MODEL_METADATA, \
    INVALID_METADATA_FIELD, SIGNATURE_NAME


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
        requested_version = request.model_spec.version.value
        valid_model_spec, version = check_availability_of_requested_model(
            models=self.models, requested_version=requested_version,
            model_name=model_name)

        if not valid_model_spec:
            context.set_code(StatusCode.NOT_FOUND)
            context.set_details(WRONG_MODEL_METADATA.format(model_name,
                                                            requested_version))
            return predict_pb2.PredictResponse()

        occurred_problem, inference_input, code = prepare_input_data(
            models=self.models, model_name=model_name, version=version,
            data=request.inputs)

        if occurred_problem:
            context.set_code(code)
            context.set_details(inference_input)
            return predict_pb2.PredictResponse()

        inference_output = self.models[model_name].engines[version] \
            .infer(inference_input)
        response = prepare_output_as_list(inference_output=inference_output,
                                          model_available_outputs=self.models
                                          [model_name].engines[version].
                                          model_keys['outputs'])
        response.model_spec.name = model_name
        response.model_spec.version.value = version
        response.model_spec.signature_name = SIGNATURE_NAME
        return response

    def GetModelMetadata(self, request, context):

        # check if model with was requested
        # is available on server with proper version
        model_name = request.model_spec.name
        requested_version = request.model_spec.version.value
        valid_model_spec, version = check_availability_of_requested_model(
            models=self.models, requested_version=requested_version,
            model_name=model_name)

        if not valid_model_spec:
            context.set_code(StatusCode.NOT_FOUND)
            context.set_details(WRONG_MODEL_METADATA.format(model_name,
                                                            requested_version))
            return get_model_metadata_pb2.GetModelMetadataResponse()

        metadata_signature_requested = request.metadata_field[0]
        if 'signature_def' != metadata_signature_requested:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(INVALID_METADATA_FIELD.format
                                (metadata_signature_requested))

            return get_model_metadata_pb2.GetModelMetadataResponse()

        inputs = self.models[model_name].engines[version].input_tensors
        outputs = self.models[model_name].engines[version].output_tensor_names

        signature_def = prepare_get_metadata_output(inputs=inputs,
                                                    outputs=outputs,
                                                    model_keys=self.models
                                                    [model_name].
                                                    engines[version].
                                                    model_keys)
        response = get_model_metadata_pb2.GetModelMetadataResponse()

        model_data_map = get_model_metadata_pb2.SignatureDefMap()
        model_data_map.signature_def['serving_default'].CopyFrom(signature_def)
        response.metadata['signature_def'].Pack(model_data_map)
        response.model_spec.name = model_name
        response.model_spec.version.value = version

        return response
