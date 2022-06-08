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
from ovmsclient import make_http_client
from utils.common import read_image_paths, load_image
from utils.resnet_utils import resnet_postprocess, get_model_io_names


parser = argparse.ArgumentParser(description='Make prediction using images in binary format')
parser.add_argument('--images_dir', required=True,
                    help='Path to a directory with images in JPG or PNG format')
parser.add_argument('--service_url', required=False, default='localhost:8000',
                    help='Specify url to http service. default:localhost:8000', dest='service_url')
parser.add_argument('--model_name', default='resnet', help='Model name to query. default: resnet',
                    dest='model_name')
parser.add_argument('--model_version', default=0, type=int, help='Model version to query. default: latest available',
                    dest='model_version')
parser.add_argument('--timeout', default=10.0, help='Request timeout. default: 10.0',
                    dest='timeout', type=float)
args = vars(parser.parse_args())

# configuration
images_dir = args.get('images_dir')
service_url = args.get('service_url')
model_name = args.get('model_name')
model_version = args.get('model_version')
timeout = args.get('timeout')

# creating http client
client = make_http_client(service_url)

# receiving metadata from model
input_name, output_name = get_model_io_names(client, model_name, model_version)

# preparing images
img_paths = read_image_paths(images_dir)

for img_path in img_paths:
    # reading image and its label
    img = load_image(img_path)

    # preparing inputs 
    inputs = {
        input_name: img
    }

    # sending predict request and receiving response
    response = client.predict(inputs, model_name, model_version, timeout)

    # response post processing
    label, _ = resnet_postprocess(response, output_name)
    print(f"Image {img_path} has been classified as {label}")
