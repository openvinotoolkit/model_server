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

import argparse
import cv2
import os
from ovmsclient.tfs_compat.grpc.serving_client import make_grpc_client
from ovmsclient.tfs_compat.grpc.requests import make_predict_request

parser = argparse.ArgumentParser(description='Make prediction using images in numpy format')
parser.add_argument('--grpc_address', required=False, default='localhost',
                    help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port', required=False, default=9000,
                    help='Specify port to grpc service. default: 9000')
parser.add_argument('--model_name', default='vehicle-detection', help='Pipeline name to query. default: detection',
                    dest='pipeline_name')
parser.add_argument('--input_name', required=False, default='image',
                    help='Specify input tensor name. default: image')
parser.add_argument('--images_input_path', required=True,
                    help='Input images path.')
parser.add_argument('--output_name', required=False, default='detection_out',
                    help='Specify output name. default: detection_out')
parser.add_argument('--output_save_path', required=False, default='./result', help='Path to store output.')
args = vars(parser.parse_args())

# configuration
address = args.get('grpc_address')
port = args.get('grpc_port')
pipeline_name = args.get('pipeline_name')
input_name = args.get('input_name')
images_input_path = args.get('images_input_path')
output_name = args.get('output_name')
output_save_path = args.get('output_save_path')


def save_image_with_detected_vehicles(image_path, output):
    image = cv2.imread(image_path)
    width = image.shape[1]
    height = image.shape[0]
    for i in range(output.shape[2]):
        if output[0, 0, i, 2] > 0.5:   # if confidence level is greater than 50%
            x_min = int(output[0, 0, i, 3] * width)
            y_min = int(output[0, 0, i, 4] * height)
            x_max = int(output[0, 0, i, 5] * width)
            y_max = int(output[0, 0, i, 6] * height)
            image = cv2.rectangle(cv2.UMat(image), (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
    cv2.imwrite(os.path.join(output_save_path, 'vehicle-detection' + '.jpg'), image)


# preparing images
with open(images_input_path) as f:
    lines = f.readlines()

# creating grpc client
config = {
    "address": address,
    "port": port
}
client = make_grpc_client(config)

for line in lines:
    # reading image
    with open(line, 'rb') as f:
        img = f.read()

    # preparing predict request
    inputs = {
        input_name: img
    }
    request = make_predict_request(inputs, pipeline_name)

    # sending predict request and receiving response
    response = client.predict(request)
    response_dict = response.to_dict()

    # output post-processing
    if output_name in response_dict.keys():
        output = response_dict[output_name]
        save_image_with_detected_vehicles(line, output)
