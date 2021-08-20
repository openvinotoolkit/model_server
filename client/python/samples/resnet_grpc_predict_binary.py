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
import numpy as np
import classes
from ovmsclient.tfs_compat.grpc.serving_client import make_grpc_client
from ovmsclient.tfs_compat.grpc.requests import make_predict_request

parser = argparse.ArgumentParser(description='Make prediction using images in binary format')
parser.add_argument('--images_list', required=False, default='/utils/resnet_images.txt',
                    help='path to image')
parser.add_argument('--grpc_address', required=False, default='localhost',
                    help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port', required=False, default=9000,
                    help='Specify port to grpc service. default: 9000')
parser.add_argument('--model_name', default='resnet', help='Model name to query. default: resnet',
                    dest='model_name')
parser.add_argument('--model_version', default=0, type=int, help='Model version to query. Lists all versions if omitted',
                    dest='model_version')
parser.add_argument('--input_name', required=False, default='0',
                    help='Specify input tensor name. default: 0')
parser.add_argument('--output_name', required=False, default='1463',
                    help='Specify output name. default: 1463')
args = vars(parser.parse_args())

# configuration
images_list = args.get('images_list')
address = args.get('grpc_address')
port = args.get('grpc_port')
model_name = args.get('model_name')
model_version = args.get('model_version')
input_name = args.get('input_name')
output_name = args.get('output_name')

# creating grpc client
config = {
    "address": address,
    "port": port
}
client = make_grpc_client(config)

# preparing images
with open(images_list) as f:
    lines = f.readlines()

image = None
label = None
for line in lines:
    # reading image and its label
    path, label = line.strip().split(" ")
    with open(path, 'rb') as f:
        image = f.read()

    # preparing predict request
    inputs = {
        input_name: image
    }
    request = make_predict_request(inputs, model_name, model_version)

    # sending predict request and receiving response
    response = client.predict(request)
    response_dict = response.to_dict()

    # output post-processing
    if output_name not in response_dict.keys():
        print(f"Invalid output name - {output_name}")
    output = response_dict[output_name]
    offset = 0
    if output.shape[1] == 1001:
        offset = 1
    ma = np.argmax(output[0]) - offset
    mark_message = ""
    if int(label) == ma:
        mark_message = "; Correct match."
    else:
        mark_message = "; Incorrect match. Should be {} {}".format(label, classes.imagenet_classes[int(label)])
    print(classes.imagenet_classes[ma], ma, mark_message)
