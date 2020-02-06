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

import datetime

from tensorflow_serving.apis import get_model_metadata_pb2
from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc, \
    model_service_pb2_grpc

from ie_serving.logger import get_logger
from ie_serving.server.constants import WRONG_MODEL_SPEC, \
    INVALID_METADATA_FIELD, SIGNATURE_NAME, GRPC
from ie_serving.server.get_model_metadata_utils import \
    prepare_get_metadata_output
from ie_serving.server.predict_utils import prepare_output, \
    prepare_input_data, StatusCode, statusCodes
from ie_serving.server.request import Request
from ie_serving.server.service_utils import \
    check_availability_of_requested_model, \
    check_availability_of_requested_status, add_status_to_response

logger = get_logger(__name__)


class PredictionServiceServicer(prediction_service_pb2_grpc.
                                PredictionServiceServicer):

    def __init__(self, models):
        self.models = models

    def Predict(self, request, context):
        """
        Predict -- provides access to loaded TensorFlow model.
        """
        # check if requested model
        # is available on server with proper version
        model_name = request.model_spec.name
        requested_version = request.model_spec.version.value
        valid_model_spec, version = check_availability_of_requested_model(
            models=self.models, requested_version=requested_version,
            model_name=model_name)

        if not valid_model_spec:
            context.set_code(StatusCode.NOT_FOUND)
            context.set_details(WRONG_MODEL_SPEC.format(model_name,
                                                        requested_version))
            logger.debug("PREDICT, invalid model spec from request, {} - {}"
                         .format(model_name, requested_version))
            return predict_pb2.PredictResponse()

        target_engine = self.models[model_name].engines[version]

        deserialization_start_time = datetime.datetime.now()
        inference_input, error_message = \
            prepare_input_data(target_engine=target_engine,
                               data=request.inputs,
                               service_type=GRPC)
        duration = (datetime.datetime.now() -
                    deserialization_start_time).total_seconds() * 1000
        logger.debug("PREDICT; input deserialization completed; {}; {}; {} ms"
                     .format(model_name, version, duration))
        if error_message is not None:
            code = statusCodes['invalid_arg'][GRPC]
            context.set_code(code)
            context.set_details(error_message)
            logger.debug("PREDICT, problem with input data. Exit code {}"
                         .format(code))
            return predict_pb2.PredictResponse()

        target_engine = self.models[model_name].engines[version]
        inference_request = Request(inference_input)
        target_engine.requests_queue.put(inference_request)
        inference_output, used_ireq_index = inference_request.wait_for_result()
        if type(inference_output) is str:
            code = statusCodes['invalid_arg'][GRPC]
            context.set_code(code)
            context.set_details(inference_output)
            logger.debug("PREDICT, problem during inference execution. Exit "
                         "code {}".format(code))
            target_engine.free_ireq_index_queue.put(used_ireq_index)
            return predict_pb2.PredictResponse()
        serialization_start_time = datetime.datetime.now()
        response = prepare_output(inference_output=inference_output,
                                  model_available_outputs=target_engine.
                                  model_keys['outputs'])
        response.model_spec.name = model_name
        response.model_spec.version.value = version
        response.model_spec.signature_name = SIGNATURE_NAME
        duration = (datetime.datetime.now() -
                    serialization_start_time).total_seconds() * 1000
        logger.debug("PREDICT; inference results serialization completed;"
                     " {}; {}; {} ms".format(model_name, version, duration))
        target_engine.free_ireq_index_queue.put(used_ireq_index)
        return response

    def GetModelMetadata(self, request, context):

        # check if model with was requested
        # is available on server with proper version
        logger.debug("MODEL_METADATA, get request: {}".format(request))
        model_name = request.model_spec.name
        requested_version = request.model_spec.version.value
        valid_model_spec, version = check_availability_of_requested_model(
            models=self.models, requested_version=requested_version,
            model_name=model_name)

        if not valid_model_spec:
            context.set_code(StatusCode.NOT_FOUND)
            context.set_details(WRONG_MODEL_SPEC.format(model_name,
                                                        requested_version))
            logger.debug("MODEL_METADATA, invalid model spec from request")
            return get_model_metadata_pb2.GetModelMetadataResponse()
        target_engine = self.models[model_name].engines[version]
        metadata_signature_requested = request.metadata_field[0]
        if 'signature_def' != metadata_signature_requested:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(INVALID_METADATA_FIELD.format
                                (metadata_signature_requested))
            logger.debug("MODEL_METADATA, invalid signature def")
            return get_model_metadata_pb2.GetModelMetadataResponse()

        inputs = target_engine.net.inputs
        outputs = target_engine.net.outputs

        signature_def = prepare_get_metadata_output(inputs=inputs,
                                                    outputs=outputs,
                                                    model_keys=target_engine.
                                                    model_keys)
        response = get_model_metadata_pb2.GetModelMetadataResponse()

        model_data_map = get_model_metadata_pb2.SignatureDefMap()
        model_data_map.signature_def['serving_default'].CopyFrom(
            signature_def)
        response.metadata['signature_def'].Pack(model_data_map)
        response.model_spec.name = model_name
        response.model_spec.version.value = version
        logger.debug("MODEL_METADATA created a response for {} - {}"
                     .format(model_name, version))
        return response


class ModelServiceServicer(model_service_pb2_grpc.ModelServiceServicer):

    def __init__(self, models):
        self.models = models

    def GetModelStatus(self, request, context):
        logger.debug("MODEL_STATUS, get request: {}".format(request))
        model_name = request.model_spec.name
        requested_version = request.model_spec.version.value
        valid_model_status = check_availability_of_requested_status(
            models=self.models, requested_version=requested_version,
            model_name=model_name)

        if not valid_model_status:
            context.set_code(StatusCode.NOT_FOUND)
            context.set_details(WRONG_MODEL_SPEC.format(model_name,
                                                        requested_version))
            logger.debug("MODEL_STATUS, invalid model spec from request")
            return get_model_status_pb2.GetModelStatusResponse()

        response = get_model_status_pb2.GetModelStatusResponse()
        if requested_version:
            version_status = self.models[model_name].versions_statuses[
                requested_version]
            add_status_to_response(version_status, response)
        else:
            for version_status in self.models[model_name].versions_statuses. \
                    values():
                add_status_to_response(version_status, response)

        logger.debug("MODEL_STATUS created a response for {} - {}"
                     .format(model_name, requested_version))
        return response
