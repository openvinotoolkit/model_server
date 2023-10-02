#
# Copyright (c) 2023 Intel Corporation
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
from tensorflow import make_tensor_proto, make_ndarray, make_tensor_proto
import datetime
import argparse
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


parser = argparse.ArgumentParser(description='Do requests to ie_serving and tf_serving using images in string format')
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--input_name',required=False, default='inputs', help='Specify input tensor name. default: inputs')
parser.add_argument('--output_name',required=False, default='outputs', help='Specify output name. default: outputs')
parser.add_argument('--model_name', default='usem', help='Define model name, must be same as is in service. default: usem')
parser.add_argument('--string',required=True, default='', help='String to query.')
args = vars(parser.parse_args())

channel = grpc.insecure_channel("{}:{}".format(args['grpc_address'],args['grpc_port']))
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


data = np.array([args['string']])
predict_request = predict_pb2.PredictRequest()
predict_request.model_spec.name = args['model_name']
predict_request.inputs[args['input_name']].CopyFrom(make_tensor_proto(data))
start_time = datetime.datetime.now()
predict_response = stub.Predict(predict_request, 10.0)
end_time = datetime.datetime.now()
duration = (end_time - start_time).total_seconds() * 1000
print("processing time", duration, "ms.")
output = make_ndarray(predict_response.outputs[args['output_name']])
print("Output shape", output.shape)
print("Output subset", output[0, :20])
