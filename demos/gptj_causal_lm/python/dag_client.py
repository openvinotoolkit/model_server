#
# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import time
import argparse
import grpc
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow import make_tensor_proto, make_ndarray

parser = argparse.ArgumentParser(description='Demo for GPT-J causal LM DAG requests using Tensorflow Serving gRPC API')

parser.add_argument('--input', required=True, help='Beginning of a sentence', type=str)
parser.add_argument('--url', required=False, help='Url to connect to', type=str, default='localhost:9000')
parser.add_argument('--model_name', required=False, help='Model name in the serving', type=str, default='my_gpt_pipeline')
args = vars(parser.parse_args())

channel = grpc.insecure_channel(args['url'])
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

input_sentence = args['input']

predict_request = predict_pb2.PredictRequest()
predict_request.model_spec.name = args['model_name']


while True:
    predict_request.inputs['texts'].CopyFrom(make_tensor_proto(np.array([input_sentence])))
    start_time = time.time()
    predict_response = stub.Predict(predict_request, 10.0)
    latency = time.time() - start_time
    results = make_ndarray(predict_response.outputs['autocompletions_string'])
    input_sentence = results[0].decode('utf-8')
    print(latency, input_sentence)
