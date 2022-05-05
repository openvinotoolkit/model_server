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

import ovmsclient
import os
import numpy as np
import argparse
import label_mapping
parser = argparse.ArgumentParser(description='Client for multi object classification pipeline')
parser.add_argument('--grpc_address', required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port', required=False, default=9178, help='Specify port to grpc service. default: 9178')
parser.add_argument('--pipeline_name', required=False, default='multi_object_classification', help='Pipeline name to request. default: multi_object_classification')
parser.add_argument('--image_input_path', required=True, help='Input image path.')

args = vars(parser.parse_args())

# Create connection to the model server
client = ovmsclient.make_grpc_client(f"{args.get('grpc_address')}:{args.get('grpc_port')}")

# Get pipeline metadata to learn about inputs
model_metadata = client.get_model_metadata(model_name=args.get('pipeline_name'))

# If model has only one input, get its name like that
input_name = next(iter(model_metadata["inputs"]))

binary_image = None
with open(args.get('image_input_path'), 'rb') as f:
    binary_image = f.read()

inputs = {input_name: binary_image}

# Run prediction and wait for the result
result = client.predict(inputs=inputs, model_name=args.get('pipeline_name'))

for batch_idx in range(result.shape[0]):
    label_id = np.argmax(result[batch_idx])
    print(f"no: {batch_idx}; label id: {label_id}; name: {label_mapping.label_mapping[label_id]}")
