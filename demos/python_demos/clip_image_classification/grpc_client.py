#
# Copyright (c) 2024 Intel Corporation
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
import sys
sys.path.append("../../common/python")
import tritonclient.grpc as grpcclient
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc
from tritonclient.utils import serialize_byte_tensor

import argparse
import datetime
import numpy as np
from client_utils import print_statistics
from urllib.request import urlretrieve
from pathlib import Path
import os
import grpc
import time

parser = argparse.ArgumentParser(description='GRPC client for clip example')

parser.add_argument('--timeout', required=False, default='15',
                    help='Specify timeout to wait for models readiness on the server in seconds. default 15 seconds.')
parser.add_argument('--url', required=False, default='localhost:9000',
                    help='Specify url to grpc service. default:localhost:9000')
parser.add_argument('--input_labels', required=False, default="cat,dog,wolf,tiger,man,horse,frog,tree,house,computer",
                    help="Specify input_labels to the CLIP model. default:cat,dog,wolf,tiger,man,horse,frog,tree,house,computer")
parser.add_argument('--image_url', required=False, default='https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg',
                    help='Specify image_url to send to the CLIP model. default:https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg')
parser.add_argument('--iterations', default=1,
                        help='Number of requests iterations, as default use number of images in numpy memmap. default: 1 ',
                        dest='iterations', type=int)

args = vars(parser.parse_args())

iterations = args.get('iterations')
iteration = 0

timeout = int(args.get('timeout'))
# Check models ready
client = grpcclient.InferenceServerClient(args['url'])
channel = grpc.insecure_channel(args['url'])
grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

while(timeout):
    request = service_pb2.ServerReadyRequest()
    response = grpc_stub.ServerReady(request)
    print("Server Ready: {}".format(response.ready))
    if response.ready:
        break
    time.sleep(1)
    timeout-=1

if not response.ready:
    print("Models are not ready. Increase timeout or check server setup and errors.")
    exit(-1)

image_url = args['image_url']
print(f"Using image_url:\n{image_url}\n")

input_name = image_url.split("/")[-1]
sample_path = Path(os.path.join("data", input_name))
if not os.path.exists(sample_path):
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(
        image_url,
        sample_path,
    )

with open(sample_path, "rb") as f:
    image_data = f.read()

image_data_bytes = serialize_byte_tensor(np.array(image_data, dtype=np.object_)).item()
image_input = grpcclient.InferInput("image", [len(image_data_bytes)], "BYTES")
image_input._raw_content = image_data_bytes

input_labels_array = args['input_labels']
input_labels = input_labels_array.split(",")
print(f"Using input_labels:\n{input_labels}\n")

labels_bytes = serialize_byte_tensor(np.array(input_labels_array, dtype=np.object_)).item()
labels_input = grpcclient.InferInput("input_labels", [len(labels_bytes)], "BYTES")
labels_input._raw_content = labels_bytes

processing_times = []
for iteration in range(iterations):
    print(f"Iteration {iteration}")
    start_time = datetime.datetime.now()
    results = client.infer("python_model", [image_input , labels_input])
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    processing_times.append(int(duration))
    print(f"Detection:\n{results.as_numpy('output_label').tobytes().decode()}\n")

print_statistics(np.array(processing_times,int), batch_size = 1)

