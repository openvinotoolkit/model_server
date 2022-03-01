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

import cv2
import os
import numpy as np
import argparse
import ovmsclient

parser = argparse.ArgumentParser(description='Client for OCR pipeline')
parser.add_argument('--grpc_address', required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port', required=False, default=9000, help='Specify port to grpc service. default: 9178')
parser.add_argument('--pipeline_name', required=False, default='face_blur_with_person_detection_pipeline', help='Pipeline name to request. default: face_blur_with_person_detection_pipeline')
parser.add_argument('--image_input_name', required=False, default='image', help='Pipeline input name for input with image. default: image')
parser.add_argument('--image_input_path', required=True, help='Input image path.')
parser.add_argument('--image_width', required=False, default=600, help='Original image width. default: 600')
parser.add_argument('--image_height', required=False, default=400, help='Original image height. default: 400')
parser.add_argument('--image_layout', required=False, default='NHWC', choices=['NCHW', 'NHWC', 'BINARY'], help='Pipeline input image layout. default: NCHW')
parser.add_argument('--blurred_image_save_path', required=False, default='', help='Path to save blurred image')
parser.add_argument('--detection_image_save_path', required=False, default='', help='Path to save image with detections')
parser.add_argument('--save_detection_with_blur', required=False, default='', help='Whether detection image should have blurred faces')

args = vars(parser.parse_args())

def load_image(path, height, width):
    img = cv2.imread(path).astype(np.float32)  # BGR color format, shape HWC
    img = cv2.resize(img, (width, height))
    return img

def prepare_input_in_nchw_format(name, path, height, width):
    img = load_image(path, height, width)
    img = img.transpose(2,0,1).reshape(1,3,height,width)
    return {name: img}

def prepare_input_in_nhwc_format(name, path, height, width):
    img = load_image(path, height, width)
    img = img.reshape(1,height,width,3)
    return {name: img}

def prepare_input_in_binary_format(name, path):
    with open(path, 'rb') as f:
        data = f.read()
        return {name: data}

def save_image(name, image, location):
    if len(image.shape) == 3 and image.shape[0] == 3:  # NCHW
        image = image.transpose(1,2,0)
    cv2.imwrite(os.path.join(location, name), image)

def save_image_with_boxes(name, image, detections, location):
    width = image.shape[1]
    height = image.shape[0]
    params = detections[0][0]
    for i in range(params.shape[0]):
        if params[i][2] > 0.7 and params[i][0] == 0:
            x_min = int(params[i][3] * width)
            y_min = int(params[i][4] * height)
            x_max = int(params[i][5] * width)
            y_max = int(params[i][6] * height)
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,0,255), 1)
    cv2.imwrite(os.path.join(location, name), image)

address = "{}:{}".format(args['grpc_address'],args['grpc_port'])

client = ovmsclient.make_grpc_client(address)

if args['image_layout'] == 'NCHW':
    input = prepare_input_in_nchw_format(args['image_input_name'], args['image_input_path'], int(args['image_height']), int(args['image_width']))
elif args['image_layout'] == 'NHWC':
    input = prepare_input_in_nhwc_format(args['image_input_name'], args['image_input_path'], int(args['image_height']), int(args['image_width']))
else:
    input = prepare_input_in_binary_format(args['image_input_name'], args['image_input_path'])

try:
    response = client.predict(input, args['pipeline_name'])
except Exception as err:
    raise err

print(f'Saving blurred image to {args["blurred_image_save_path"]}/blurred_image.jpg')
save_image('blurred_image.jpg', response['image'][0], args['blurred_image_save_path'])

if 'detection' in response.keys() and args['detection_image_save_path']:
    print(f'Saving image with detections to {args["detection_image_save_path"]}/detections_image.jpg')
    if args['save_detection_with_blur']:
        print('Blur effect will be combined with detections')
        img = response['image'][0]
    else:
        img = load_image(args['image_input_path'], int(args['image_height']), int(args['image_width']))
    save_image_with_boxes('detections_image.jpg', img, response['detection'], args['detection_image_save_path'])
else:
    print('Skipping saving image with detections since no detection_image_save_path provided')
