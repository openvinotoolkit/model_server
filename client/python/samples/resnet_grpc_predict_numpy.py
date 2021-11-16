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

import numpy as np
import argparse
from ovmsclient import make_grpc_client, make_grpc_predict_request
from utils.common import get_model_io_names
from utils.resnet_utils import resnet_postprocess


parser = argparse.ArgumentParser(description='Make prediction using images in numerical format')
parser.add_argument('--images_numpy', required=True,
                    help='Path to a .npy file with data to infer')
parser.add_argument('--service_url', required=False, default='localhost:9000',
                    help='Specify url to grpc service. default:localhost', dest='service_url')
parser.add_argument('--model_name', default='resnet', help='Model name to query. default: resnet',
                    dest='model_name')
parser.add_argument('--model_version', default=0, type=int, help='Model version to query. default: latest available',
                    dest='model_version')
parser.add_argument('--iterations', default=0, type=int, help='Total number of requests to be sent. default: 0 - all elements in numpy',
                    dest='iterations')
args = vars(parser.parse_args())

# configuration
images_numpy_path = args.get('images_numpy')
service_url = args.get('service_url')
model_name = args.get('model_name')
model_version = args.get('model_version')
iterations = args.get('iterations')

# creating grpc client
client = make_grpc_client(service_url)

# receiving metadata from model
input_name, output_name = get_model_io_names(client, model_name, model_version)

# preparing images
imgs = np.load(images_numpy_path, mmap_mode='r', allow_pickle=False)

if iterations == 0:
    iterations = imgs.shape[0]

for i in range (iterations):
    # preparing predict request
    inputs = {
        input_name: [imgs[i%imgs.shape[0]]]
    }
    request = make_grpc_predict_request(inputs, model_name, model_version)

    # sending predict request and receiving response
    response = client.predict(request)

    # response post processing
    label, confidence_score = resnet_postprocess(response, output_name)
    print(f"Image #{i%imgs.shape[0]} has been classified as {label} with {confidence_score*100}% confidence")
