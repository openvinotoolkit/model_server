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
from tensorflow import make_tensor_proto, make_ndarray
import numpy as np


class ModelNotFoundError(Exception):
    pass


class RequestProcessingError(Exception):
    pass


class OvmsUnavailableError(Exception):
    pass


class OvmsConnector():
    def __init__(self, ovms_port, ovms_model_info):
        self.ovms_port = ovms_port
        self.model_name = ovms_model_info['model_name']
        self.model_version = ovms_model_info['model_version']
        self.input_name = ovms_model_info['input_name']

        channel = grpc.insecure_channel("{}:{}".format("127.0.0.1", self.ovms_port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    def send(self, inference_input):
        if inference_input is None:
            raise ValueError("Invalid inference input")

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.version.value = int(self.model_version)
        try:
            tensor_proto = make_tensor_proto(inference_input, shape=(inference_input.shape))
        except TypeError as e:
            raise TypeError("Unsupported data type") from e
        except ValueError as e:
            raise ValueError("Invalid arguments") from e
        request.inputs[self.input_name].CopyFrom(tensor_proto)
        try:
            result = self.stub.Predict(request, 10.0)
            return_dict = {}
            for output in result.outputs:
                return_dict[output] = make_ndarray(result.outputs[output])
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise RequestProcessingError("Error during inference request \
                        processing: {}".format(e.details())) from e
            elif e.code() == grpc.StatusCode.NOT_FOUND:
                raise ModelNotFoundError("Requested model not found") from e
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                raise OvmsUnavailableError("Unable to connect to OVMS") from e
            else:
                raise Exception("GRPC error") from e
        except TypeError as e:
            raise TypeError("Output datatype error") from e

        return return_dict

