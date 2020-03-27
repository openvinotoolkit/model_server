#!/usr/bin/env python3
#
# Copyright (c) 2018-2020 Intel Corporation
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

import sys
import grpc
import datetime
import argparse
import numpy as np
from tensorflow import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


parser = argparse.ArgumentParser(
    description='Sends requests via TFS gRPC API using images in numpy format.'
                ' It measures performance statistics.')

parser.add_argument('--images_numpy_path',
                    required=True,
                    help='image in numpy format')
parser.add_argument('--grpc_address',
                    required=False,
                    default='localhost',
                    help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port',
                    required=False,
                    default=9178,
                    help='Specify port to grpc service. default: 9000')
parser.add_argument('--input_name',
                    required=False,
                    default='input',
                    help='Specify input tensor name. default: input')
parser.add_argument('--iterations',
                    default=0,
                    help='Number of requests iterations, '
                    'as default use number of images in numpy memmap. '
                    'default: 0 (consume all frames)',
                    type=int)
parser.add_argument('--batchsize',
                    default=1,
                    help='Number of images in a single request. default: 1',
                    type=int)
parser.add_argument('--model_name',
                    default='resnet',
                    help='Define model name in payload. default: resnet')
parser.add_argument('--report_every',
                    default=0,
                    help='Report performance every X iterations',
                    type=int)
parser.add_argument('--id',
                    default='--',
                    help='Helps identifying client')
args = parser.parse_args()

channel = grpc.insecure_channel("{}:{}".format(
    args.grpc_address,
    args.grpc_port))

stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

processing_times = np.zeros((0), int)

imgs = np.load(args.images_numpy_path, mmap_mode='r', allow_pickle=False)
imgs = imgs - np.min(imgs)  # Normalization 0-255
imgs = imgs / np.ptp(imgs) * 255  # Normalization 0-255

# If input numpy file has too few frames according to the
# value of iterations and the batch size,
# it will be duplicated to match requested number of frames.
while args.batchsize >= imgs.shape[0]:
    imgs = np.append(imgs, imgs, axis=0)

if args.iterations < 0:
    print("Argument '--iterations' can't be lower than 0")
    print("Exitting")
    sys.exit(1)
elif args.iterations == 0:
    iterations = int(imgs.shape[0] // args.batchsize)
else:
    iterations = args.iterations

iteration = 0

print("[{:2}] Starting iterations".format(args.id))

while iteration <= iterations:
    for x in range(0, imgs.shape[0] - args.batchsize + 1, args.batchsize):
        iteration += 1
        if iteration > iterations:
            break

        # Preparing image data
        img = imgs[x:(x + args.batchsize)]

        # Creating request object
        request = predict_pb2.PredictRequest()
        request.model_spec.name = args.model_name

        # Populating request with data
        request.inputs[args.input_name].CopyFrom(
            make_tensor_proto(img, shape=(img.shape)))

        # Measuring gRPC request time
        start_time = datetime.datetime.now()
        result = stub.Predict(request, 10.0)
        end_time = datetime.datetime.now()

        # Aggregating processing time statistics
        duration = (end_time - start_time).total_seconds() * 1000
        processing_times = np.append(processing_times, np.array([duration]))

        if args.report_every > 0 and iteration < iterations and iteration % args.report_every == 0:
            print(f'[{args.id:2}] Iteration {iteration:5}/{iterations:5}; '
                  f'Current latency: {round(duration, 2):.2f}ms; '
                  f'Average latency: {round(np.average(processing_times), 2):.2f}ms')

print(f"[{args.id:2}] "
      f"Iterations: {iterations:5}; "
      f"Final average latency: {round(np.average(processing_times), 2):.2f}ms")
