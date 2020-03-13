#
# Copyright (c) 2020 Intel Corporation
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
from abc import ABC, abstractmethod
import ie_serving.messaging.apis.endpoint_requests_pb2 as ovms_ipc
from ie_serving.messaging.apis.endpoint_responses_pb2 \
    import EndpointResponse, PredictResponse
from ie_serving.messaging.apis.data_attributes_pb2 import NumpyAttributes
import numpy as np

import zmq
import multiprocessing.shared_memory

from threading import Thread


class Engine(ABC):

    def __init__(self, engine_properties):
        self.model_name = engine_properties["model_name"]
        self.model_version = engine_properties["model_version"]
        # engine properties should be a dict that will
        # be used by build_engine method
        self.build_engine(engine_properties)

        self.socket_name = self.model_name + '-' + str(self.model_version)
        self.dispatcher = Thread(
            target=self.prediction_listener, args=(self.socket_name,))
        self.dispatcher.start()
        self.dispatcher.join()

    @abstractmethod
    def build_engine(self, engine_properties):
        pass

    @abstractmethod
    def predict(self, data, return_socket_name):
        pass

    def prediction_listener(self, socket_name):
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind(
            "ipc:///tmp/{}.sock".format(socket_name))
        while True:
            req = self.zmq_socket.recv()
            self.zmq_socket.send(b'ACK')
            data = {}
            request = ovms_ipc.EndpointRequest()
            request.MergeFromString(req)
            if not request.HasField("predict_request"):
                continue
            for inference_input in request.predict_request.inputs:
                shm = multiprocessing.shared_memory.SharedMemory(
                    name=inference_input.shm_name)
                data[inference_input.input_name] = np.ndarray(
                    tuple(inference_input.numpy_attributes.shape),
                    dtype=inference_input.numpy_attributes.data_type, buffer=shm.buf)
            return_socket_name = request.predict_request.return_socket_name
            self.predict(data, return_socket_name)

    def return_results(self, inference_output, return_socket_name):
        zmq_return_context = zmq.Context()
        zmq_return_socket = zmq_return_context.socket(zmq.REQ)
        zmq_return_socket.connect(
            "ipc://{}".format(return_socket_name))
        ipc_endpoint_response = EndpointResponse()
        ipc_predict_response = PredictResponse()
        ipc_outputs = []

        for output_name in list(inference_output.keys()):
            single_output = inference_output[output_name]
            output_shm = multiprocessing.shared_memory.SharedMemory(create=True,
                                                                    size=single_output.nbytes)
            shm_array = np.ndarray(single_output.shape, dtype=single_output.dtype,
                                   buffer=output_shm.buf)
            shm_array[:] = single_output[:]

            ipc_numpy_attributes = NumpyAttributes()
            ipc_numpy_attributes.shape.extend(list(shm_array.shape))
            ipc_numpy_attributes.data_type = shm_array.dtype.name

            ipc_output_data = PredictResponse.Data()
            ipc_output_data.numpy_attributes.CopyFrom(ipc_numpy_attributes)
            ipc_output_data.output_name = output_name
            ipc_output_data.shm_name = output_shm.name
            ipc_outputs.append(ipc_output_data)

        ipc_predict_response.outputs.extend(ipc_outputs)
        ipc_predict_response.responding_version = 1
        ipc_endpoint_response.predict_response.CopyFrom(ipc_predict_response)
        msg = ipc_endpoint_response.SerializeToString()
        zmq_return_socket.send(msg)
        zmq_return_socket.recv()

