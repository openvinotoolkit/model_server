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
import cv2
import os
import numpy as np
from tensorflow import make_tensor_proto, make_ndarray
import argparse
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
parser = argparse.ArgumentParser(description='Client for image transformation node testing')
parser.add_argument('--grpc_address', required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port', required=False, default=9178, help='Specify port to grpc service. default: 9178')
parser.add_argument('--pipeline_name', required=False, default='image_transformation_test', help='Pipeline name to request. default: image_transformation_test')
parser.add_argument('--image_input_name', required=False, default='image', help='Pipeline input name for input with image. default: image')
parser.add_argument('--image_output_name', required=False, default='image', help='Pipeline out name for input with image. default: image')
parser.add_argument('--input_image_path', required=True, help='Location to load input image.')
parser.add_argument('--output_image_path', required=True, help='Location where to save output image.')
parser.add_argument('--image_width', required=False, default=1920, help='Reshape before sending to given image width. default: 1920')
parser.add_argument('--image_height', required=False, default=1024, help='Reshape before sending to given image height. default: 1024')
parser.add_argument('--input_layout', required=False, default='NCHW', choices=['NCHW', 'NHWC', 'BINARY'], help='Input image layout. default: NCHW')
parser.add_argument('--input_color', required=False, default='BGR', choices=['BGR', 'RGB', 'GRAY'], help='Input image color order. default: BGR')
parser.add_argument('--output_layout', required=False, default='NCHW', choices=['NCHW', 'NHWC'], help='Output image layout. default: NCHW')
parser.add_argument('--output_color', required=False, default='BGR', choices=['BGR', 'RGB', 'GRAY'], help='Output image color order. default: BGR')
parser.add_argument('--scale_values', required=False, default=None, nargs='*', type=float, help='scale image values before sending')
parser.add_argument('--mean_values', required=False, default=None, nargs='*', type=float, help='mean image values before sending')

args = vars(parser.parse_args())

def scale(img, values):
    channels = cv2.split(img)
    number_of_channels = img.shape[-1]
    if len(values) != number_of_channels:
        print('scale values must match number of channels')
        exit(1)
    i = 0
    for i in range(number_of_channels):
        channels[i] /= values[i]
    img = cv2.merge(channels)
    img = img.reshape(img.shape[0], img.shape[1], number_of_channels)
    return img

def mean(img, values):
    channels = cv2.split(img)
    number_of_channels = img.shape[-1]
    if len(values) != number_of_channels:
        print('mean values must match number of channels')
        exit(1)
    i = 0
    for i in range(number_of_channels):
        channels[i] -= values[i]
    img = cv2.merge(channels)
    img = img.reshape(img.shape[0], img.shape[1], number_of_channels)
    return img

def prepare_img_input(request, name, path, width, height, layout, color):
    if layout == 'BINARY':
        with open(path, 'rb') as f:
            request.inputs[name].CopyFrom(make_tensor_proto([f.read()], shape=[1]))
        return

    img = cv2.imread(path).astype(np.float32)  # BGR color format, shape HWC
    img = cv2.resize(img, (width, height))
    if color == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color == 'GRAY':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape
        img = img.reshape(h, w, 1)

    if args['scale_values']:
        img = scale(img, args['scale_values'])

    if args['mean_values']:
        img = mean(img, args['mean_values'])

    if layout == 'NCHW':
        h, w, c = img.shape
        img = img.transpose(2,0,1).reshape(1, c, h, w)
    else:
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

    request.inputs[name].CopyFrom(make_tensor_proto(img, shape=img.shape))

def save_img_output_as_jpg(output_nd, path, layout, color):
    img = output_nd[0]
    if layout == 'NCHW':
        img = img.transpose(1,2,0)
    if color == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

address = "{}:{}".format(args['grpc_address'],args['grpc_port'])
MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024
channel = grpc.insecure_channel(address,
    options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ])

stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = args['pipeline_name']

prepare_img_input(
    request,
    args['image_input_name'],
    args['input_image_path'],
    int(args['image_width']),
    int(args['image_height']),
    args['input_layout'],
    args['input_color'])

response = stub.Predict(request, 30.0)

for name in response.outputs:
    print(f"Output: name[{name}]")
    tensor_proto = response.outputs[name]
    output_nd = make_ndarray(tensor_proto)
    print(f"    numpy => shape[{output_nd.shape}] data[{output_nd.dtype}]")

    if name == args['image_output_name']:
        save_img_output_as_jpg(output_nd, args['output_image_path'], args['output_layout'], args['output_color'])
