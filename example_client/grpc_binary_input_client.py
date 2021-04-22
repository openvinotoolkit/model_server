#
# Copyright (c) 2021 Intel Corporation
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
import requests
import tensorflow as tf
import argparse
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


parser = argparse.ArgumentParser(description='Sends requests via TFS gRPC API using images in binary format.')
parser.add_argument('--image_path', required=False, help='Image in binary format. default: https://tensorflow.org/images/blogs/serving/cat.jpg')
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default: localhost')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--input_name',required=False, default='input', help='Specify input tensor name. default: input')
parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. default: resnet',
                    dest='model_name')

args = vars(parser.parse_args())
# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

if args['image_path']:
  with open(args['image_path'], 'rb') as f:
    data = f.read()
else:
  # Download the image since we weren't given one
  dl_request = requests.get(IMAGE_URL, stream=True)
  dl_request.raise_for_status()
  data = dl_request.content

address = "{}:{}".format(args['grpc_address'],args['grpc_port'])
channel = grpc.insecure_channel(address)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
# Send request
# See prediction_service.proto for gRPC request/response details.
request = predict_pb2.PredictRequest()
request.model_spec.name = args['model_name']
request.inputs[args['input_name']].CopyFrom(
    tf.make_tensor_proto(data, shape=[1]))
result = stub.Predict(request, 10.0)  # 10 secs timeout
print(result)
