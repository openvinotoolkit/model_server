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

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

class OvmsConnector():
    def __init__(self, ovms_port, ovms_model_info):
        self.ovms_port = ovms_port
        self.model_name = ovms_model_info['model_name']
        self.model_version = ovms_model_info['model_version']
        self.input_name = ovms_model_info['input_name']
        self.input_shape = ovms_model_info['input_shape']

        channel = grpc.insecure_channel("{}:{}".format("127.0.0.1", self.ovms_port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    
    def send(self, inference_input):
        # TODO: prepare request and handle response
        request = predict_pb2.PredictRequest()
        result = self.stub.Predict(request, 10.0)
        return 
