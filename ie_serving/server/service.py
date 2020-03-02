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
import os
import threading

import zmq
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc, \
    model_service_pb2_grpc

from ie_serving.config import GLOBAL_CONFIG
from ie_serving.logger import get_logger
from ie_serving.messaging.endpoint_responses_pb2 import EndpointResponse
from ie_serving.server.constants import WRONG_MODEL_SPEC, SIGNATURE_NAME
from ie_serving.server.predict_utils import prepare_output, StatusCode
from ie_serving.messaging.predict_msg_processing import \
    prepare_ipc_predict_request
from ie_serving.messaging.msg_processing import extract_ipc_response
from ie_serving.shm_management import free_outputs_shm

import numpy as np
from multiprocessing import shared_memory
logger = get_logger(__name__)


class PredictionServiceServicer(prediction_service_pb2_grpc.
                                PredictionServiceServicer):

    def __init__(self):
        self.zmq_context = zmq.Context()
        self.process_id = os.getpid()
        self.predict_return_sockets = {}

    def Predict(self, request, context):
        # check if requested model
        # is available on server with proper version
        model_name = request.model_spec.name
        requested_version = request.model_spec.version.value

        start_time = datetime.datetime.now()
        if requested_version == 0:
            # requested_version = "default"
            requested_version = 1
        target_socket_name = os.path.join(GLOBAL_CONFIG['tmp_files_dir'],
                                          "{}-{}.sock".format(
                                              model_name, requested_version))
        if not os.path.exists(target_socket_name):
            context.set_code(StatusCode.NOT_FOUND)
            context.set_details(WRONG_MODEL_SPEC.format(model_name,
                                                        requested_version))
            logger.debug("PREDICT, invalid model spec from request, {} - {}"
                         .format(model_name, requested_version))
            return predict_pb2.PredictResponse()
        duration = (datetime.datetime.now()
                    - start_time).total_seconds() * 1000
        logger.debug("Version existence check - {} ms".format(duration))

        start_time = datetime.datetime.now()
        target_socket = self.zmq_context.socket(zmq.REQ)
        target_socket.connect("ipc://{}".format(target_socket_name))
        duration = (datetime.datetime.now()
                    - start_time).total_seconds() * 1000
        logger.debug("Engine connection setup - {} ms".format(duration))

        thread_id = threading.get_ident()
        return_socket_name = os.path.join(GLOBAL_CONFIG['tmp_files_dir'],
                                          "{}-{}.sock".format(self.process_id,
                                                              thread_id))
        start_time = datetime.datetime.now()
        ipc_predict_request = prepare_ipc_predict_request(
            None, data=request.inputs, return_socket_name=return_socket_name)
        duration = (datetime.datetime.now()
                    - start_time).total_seconds() * 1000
        logger.debug("Preparing IPC message - {} ms"
                     .format(duration))

        start_time = datetime.datetime.now()
        target_socket.send(ipc_predict_request.SerializeToString())
        target_socket.recv()
        duration = (datetime.datetime.now()
                    - start_time).total_seconds() * 1000
        logger.debug("Sending IPC message and receiving confirmation - {} ms"
                     .format(duration))

        if return_socket_name not in self.predict_return_sockets:
            return_socket = self.zmq_context.socket(zmq.REP)
            return_socket.bind("ipc://{}".format(return_socket_name))
            self.predict_return_sockets[return_socket_name] = return_socket

        return_socket = self.predict_return_sockets[return_socket_name]
        logger.debug("Awaiting return IPC message")
        start_time = datetime.datetime.now()
        ipc_raw_response = return_socket.recv()
        duration = (datetime.datetime.now()
                    - start_time).total_seconds() * 1000
        logger.debug("Backend processing and communication time - {} ms"
                     .format(duration))
        return_socket.send(b'ACK')

        start_time = datetime.datetime.now()
        ipc_endpoint_response = EndpointResponse()
        ipc_endpoint_response.MergeFromString(ipc_raw_response)
        ipc_predict_response, status = extract_ipc_response(
            ipc_endpoint_response, "predict_response")
        duration = (datetime.datetime.now()
                    - start_time).total_seconds() * 1000
        logger.debug("IPC response unpacking - {} ms".format(duration))

        if status['error_code'] != StatusCode.OK:
            context.set_code(status['error_code'])
            context.set_details(status['error_message'])
            return predict_pb2.PredictResponse()

        start_time = datetime.datetime.now()
        inference_output = {}
        for output in ipc_predict_response.outputs:
            output_shm = shared_memory.SharedMemory(name=output.shm_name)
            output_results = np.ndarray(
                shape=tuple(output.numpy_attributes.shape),
                dtype=np.dtype(output.numpy_attributes.data_type),
                buffer=output_shm.buf)
            inference_output[output.output_name] = output_results
        # inference_output = extract_inference_output(ipc_predict_response)
        duration = (datetime.datetime.now()
                    - start_time).total_seconds() * 1000
        logger.debug("Preparing output - {} ms".format(duration))

        start_time = datetime.datetime.now()
        response = prepare_output(inference_output=inference_output)
        duration = (datetime.datetime.now()
                    - start_time).total_seconds() * 1000
        logger.debug("Output serialization - {} ms".format(duration))

        response.model_spec.name = model_name
        response.model_spec.version.value = ipc_predict_response.\
            responding_version
        response.model_spec.signature_name = SIGNATURE_NAME
        free_outputs_shm(ipc_predict_response)
        return response

    # GetModelMetadata and GetModelStatus endpoints will be
    # enabled in future development phases


"""
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
"""


class ModelServiceServicer(model_service_pb2_grpc.ModelServiceServicer):

    def __init__(self):
        pass
    """

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
    """
