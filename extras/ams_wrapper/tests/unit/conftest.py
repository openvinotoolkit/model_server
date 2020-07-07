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
import numpy as np
from tensorflow import make_tensor_proto


class FakeGrpcError(grpc.RpcError):
    def __init__(self, error_code):
        self.error_code = error_code

    def code(self):
        return self.error_code

    def details(self):
        return "THIS IS FAKE GRPC ERROR - FOR TESTING PURPOSES ONLY"


class FakeGrpcPredictResponse():
    def __init__(self, output_dict):
        self.outputs = output_dict


class FakeGrpcStub:
    def __init__(self, error_code):
        self.error_code = error_code

    def Predict(self, request, timeout):
        if self.error_code is None:
            return FakeGrpcPredictResponse({"output": make_tensor_proto(np.zeros((1, 1000)))})
        raise FakeGrpcError(self.error_code)
