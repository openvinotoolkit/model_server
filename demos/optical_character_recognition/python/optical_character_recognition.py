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

parser = argparse.ArgumentParser(description='Client for OCR pipeline')
parser.add_argument('--grpc_address', required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port', required=False, default=9178, help='Specify port to grpc service. default: 9178')
parser.add_argument('--pipeline_name', required=False, default='ocr', help='Pipeline name to request. default: ocr')
parser.add_argument('--image_input_name', required=False, default='image', help='Pipeline input name for input with image. default: image')
parser.add_argument('--image_input_path', required=True, help='Input image path.')
parser.add_argument('--texts_output_name', required=False, default='texts', help='Pipeline output name for output with recognized texts. default: texts')
parser.add_argument('--text_images_output_name', required=False, default='text_images', help='Pipeline output name for cropped images with text. default: text_images')
parser.add_argument('--text_images_save_path', required=False, default='', help='If specified, images will be saved to disk.')
parser.add_argument('--image_width', required=False, default=1920, help='Original image width. default: 1920')
parser.add_argument('--image_height', required=False, default=1024, help='Original image height. default: 1024')
parser.add_argument('--image_layout', required=False, default='NCHW', choices=['NCHW', 'NHWC', 'BINARY'], help='Pipeline input image layout. default: NCHW')

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

def save_text_images_as_jpgs(output_nd, name, location):
    for i in range(output_nd.shape[0]):
        out = output_nd[i][0]
        if len(out.shape) == 3 and out.shape[0] == 3:  # NCHW
            out = out.transpose(1,2,0)
        cv2.imwrite(os.path.join(location, name + '_' + str(i) + '.jpg'), out)

def decode(text):
    word = ''
    last_character = None
    for character in text:
        if character == last_character:
            continue
        elif character == '#':
            last_character = None
        else:
            last_character = character
            word += character
    return word

def text_recognition_output_to_text(output_nd):
    for i in range(output_nd.shape[0]):
        data = output_nd[i]
        alphabet = '0123456789abcdefghijklmnopqrstuvwxyz#'
        preds = data.argmax(2)
        word = ''
        for i in range(preds.shape[0]):
            word += alphabet[preds[i,0]]
        print(decode(word))


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
        print('No text has been found in the image')
        exit(1)
    else:
        raise err

for name in response.outputs:
    print(f"Output: name[{name}]")
    tensor_proto = response.outputs[name]
    output_nd = make_ndarray(tensor_proto)
    print(f"    numpy => shape[{output_nd.shape}] data[{output_nd.dtype}]")
    if name == args['text_images_output_name'] and len(args['text_images_save_path']) > 0:
        save_text_images_as_jpgs(output_nd, name, args['text_images_save_path'])
    if name == args['texts_output_name']:
        text_recognition_output_to_text(output_nd)