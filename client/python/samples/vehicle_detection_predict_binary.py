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
from utils.common import load_image, read_image_paths, get_model_io_names
from utils.vehicle_utils import vehicle_postprocess

parser = argparse.ArgumentParser(description='Make vehicle detection prediction using images in binary format')
parser.add_argument('--images_dir', required=True,
                    help='Path to a directory with images in JPG or PNG format')
parser.add_argument('--grpc_address', required=False, default='localhost',
                    help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port', required=False, default=9000, type=int,
                    help='Specify port to grpc service. default: 9000')
parser.add_argument('--model_name', default='vehicle-detection',
                    help='Model name to query. default: vehicle-detection')
parser.add_argument('--model_version', default=0, type=int,
                    help='Model version to query. default: latest available')
parser.add_argument('--output_save_path', required=True,
                    help='Path to store output.')
args = vars(parser.parse_args())

# configuration
images_dir = args.get('images_dir')
address = args.get('grpc_address')
port = args.get('grpc_port')
model_name = args.get('model_name')
model_version = args.get('model_version')
output_save_path = args.get('output_save_path')

# preparing images
img_paths = read_image_paths(images_dir)

# creating grpc client
config = {
    "address": address,
    "port": port
}
client = make_grpc_client(config)

# receiving metadata from model
input_name, output_name = get_model_io_names(client, model_name, model_version)

for img_path in img_paths:
    # reading image and its label
    img_path = img_path.strip()
    img = load_image(img_path)

    # preparing predict request
    inputs = {
        input_name: img
    }
    request = make_grpc_predict_request(inputs, model_name)

    # sending predict request and receiving response
    response = client.predict(request)

    # output post-processing
    vehicle_postprocess(response, img_path, output_name, output_save_path)
