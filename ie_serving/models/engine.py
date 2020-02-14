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
import ie_serving.messaging.endpoint_requests_pb2 as ovms_ipc

import zmq
import multiprocessing


class Engine(ABC):

    def __init__(self, model_name, model_version, net, plugin,
                 mapping_config, exec_net, batching_info, shape_info,
                 free_ireq_index_queue, num_ireq, requests_queue,
                 target_device, plugin_config):
        self.model_name = model_name
        self.model_version = model_version
        self.exec_net = exec_net
        self.net = net
        self.batching_info = batching_info
        self.shape_info = shape_info
        self.plugin = plugin

        self.free_ireq_index_queue = free_ireq_index_queue
        self.num_ireq = num_ireq
        self.requests_queue = requests_queue

        self.target_device = target_device
        self.plugin_config = plugin_config

        self.engine_active = True
        self.socket_name = model_name + model_version
        self.dispatcher = multiprocessing.Process(
            target=self.prediction_listener, args=(self.socket_name,))
        self.dispatcher.start()
        self.dispatcher.join()

    def prediction_listener(self, socket_name):
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind(
            "ipc:///tmp/{}.sock".format(socket_name))
        while True:
            req = self.zmq_socket.recv()
            self.zmq_socket.send(b'ACK')
            data = self.parse_request(req)
            if data:
                self.run_inference(data)

    def management_listener(self):
        pass

    def parse_request(self, req):
        data = {}
        request = ovms_ipc.EndpointRequest()
        request.ParseFromString(req)
        if not request.HasField("predict_request"):
            return None
        for inference_input in request.predict_request.inputs:
            data[inference_input.input_name] = inference_input.shm_name
        return data

    @abstractmethod
    def run_inference(self, data):
        return
