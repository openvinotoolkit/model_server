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
from ovmsclient import make_grpc_client, make_grpc_predict_request
from utils.common import get_model_io_names, read_imgs_as_ndarray, get_model_input_shape
from utils.resnet_utils import resnet_postprocess
from imagenet_classes import imagenet_classes


parser = argparse.ArgumentParser(description='Make prediction using images in binary format')
parser.add_argument('--images_dir', required=True,
                    help='Path to a directory with images in JPG or PNG format')
parser.add_argument('--grpc_address', required=False, default='localhost',
                    help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port', required=False, default=9000,
                    help='Specify port to grpc service. default: 9000')
parser.add_argument('--model_name', default='resnet', help='Model name to query. default: resnet',
                    dest='model_name')
parser.add_argument('--model_version', default=0, type=int, help='Model version to query. default: latest available',
                    dest='model_version')
args = vars(parser.parse_args())

# configuration
images_dir = args.get('images_dir')
address = args.get('grpc_address')
port = args.get('grpc_port')
model_name = args.get('model_name')
model_version = args.get('model_version')

# creating grpc client
config = {
    "address": address,
    "port": port
}
client = make_grpc_client(config)

# receiving metadata from model
input_name, output_name = get_model_io_names(client, model_name, model_version)
input_shape = get_model_input_shape(client, model_name, model_version)
input_layout = "NHWC"

# preparing images
imgs = read_imgs_as_ndarray(images_dir, input_shape, input_layout)

for i, img in enumerate(imgs):
    # preparing predict request
    inputs = {
        input_name: [img]
    }
    request = make_grpc_predict_request(inputs, model_name, model_version)

    # sending predict request and receiving response
    response = client.predict(request)

    # response post processing
    label, confidence_score = resnet_postprocess(response, output_name)
    print(f"Image {i+1} has been classified as {imagenet_classes[label]} with {confidence_score}% confidence")
