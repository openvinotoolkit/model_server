#
# Copyright (c) 2022 Intel Corporation
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

parser = argparse.ArgumentParser(description='Client for OCR pipeline')
parser.add_argument('--grpc_address', required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port', required=False, default=9000, help='Specify port to grpc service. default: 9178')
parser.add_argument('--pipeline_name', required=False, default='face_blur_pipeline', help='Pipeline name to request. default: face_blur_pipeline')
parser.add_argument('--image_input_name', required=False, default='image', help='Pipeline input name for input with image. default: image')
parser.add_argument('--image_input_path', required=True, help='Input image path.')
parser.add_argument('--people_images_output_name', required=False, default='people_images', help='Pipeline output name for cropped images with people. default: people_images')
parser.add_argument('--people_images_save_path', required=False, default='', help='If specified, people images will be saved to disk.')
parser.add_argument('--image_width', required=False, default=600, help='Original image width. default: 600')
parser.add_argument('--image_height', required=False, default=400, help='Original image height. default: 400')
parser.add_argument('--image_layout', required=False, default='NHWC', choices=['NCHW', 'NHWC', 'BINARY'], help='Pipeline input image layout. default: NCHW')
parser.add_argument('--image_output_name', required=False, default='image', help='Pipeline output name for output with image. default: image')
parser.add_argument('--detection_output_name', required=False, default='detection', help='Pipeline output name for output with detection. default: detection')

args = vars(parser.parse_args())

def prepare_img_input_in_nchw_format(request, name, path, resize_to_shape):
    img = cv2.imread(path).astype(np.float32)  # BGR color format, shape HWC
    img = cv2.resize(img, (resize_to_shape[1], resize_to_shape[0]))
    target_shape = (img.shape[0], img.shape[1])
    img = img.transpose(2,0,1).reshape(1,3,target_shape[0],target_shape[1])
    request.inputs[name].CopyFrom(make_tensor_proto(img, shape=img.shape))

def prepare_img_input_in_nhwc_format(request, name, path, resize_to_shape):
    img = cv2.imread(path).astype(np.float32)  # BGR color format, shape HWC
    img = cv2.resize(img, (resize_to_shape[1], resize_to_shape[0]))
    target_shape = (img.shape[0], img.shape[1])
    img = img.reshape(1,target_shape[0],target_shape[1],3)
    request.inputs[name].CopyFrom(make_tensor_proto(img, shape=img.shape))

def prepare_img_input_in_binary_format(request, name, path):
    with open(path, 'rb') as f:
        data = f.read()
        request.inputs[name].CopyFrom(make_tensor_proto(data, shape=[1]))

def draw_boxes(image, detections):
    width = image.shape[2] if image.shape[0] == 3 else image.shape[1]
    height = image.shape[1] if image.shape[0] == 3 else image.shape[0]
    params = detections[0][0]
    for i in range(params.shape[0]):
        if params[i][2] > 0.7 and params[i][0] == 0:
            x_min = int(params[i][3] * width)
            y_min = int(params[i][4] * height)
            x_max = int(params[i][5] * width)
            y_max = int(params[i][6] * height)
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,0,255), 1)
    return image

def save_image(image, location):
    if len(image.shape) == 3 and image.shape[0] == 3:  # NCHW
        image = image.transpose(1,2,0)
    cv2.imwrite(os.path.join(location, 'image.jpg'), image)

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

if args['image_layout'] == 'NCHW':
    prepare_img_input_in_nchw_format(request, args['image_input_name'], args['image_input_path'], (int(args['image_height']), int(args['image_width'])))
elif args['image_layout'] == 'NHWC':
    prepare_img_input_in_nhwc_format(request, args['image_input_name'], args['image_input_path'], (int(args['image_height']), int(args['image_width'])))
else:
    prepare_img_input_in_binary_format(request, args['image_input_name'], args['image_input_path'])

try:
    response = stub.Predict(request, 30.0)
except grpc.RpcError as err:
    if err.code() == grpc.StatusCode.ABORTED:
        print('Nothing has been found in the image')
        exit(1)
    else:
        raise err

if args['image_output_name'] in response.outputs.keys():
    blurred_image = make_ndarray(response.outputs[args['image_output_name']])[0]
else:
    print(f'No output found in {args["image_output_name"]}')
    print('Available outputs:')
    for name in response.outputs:
        print(name)
    exit(1)

for name in response.outputs:
    print(f"Output: name[{name}]")
    tensor_proto = response.outputs[name]
    output_nd = make_ndarray(tensor_proto)
    print(f"    numpy => shape[{output_nd.shape}] data[{output_nd.dtype}]")
    
    if name == args['detection_output_name']:
        blurred_image = draw_boxes(blurred_image, output_nd)
        save_image(blurred_image, args['people_images_save_path'])
