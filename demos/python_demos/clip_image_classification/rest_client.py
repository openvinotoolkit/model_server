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
import tritonclient.http as httpclient
import argparse
import datetime
import numpy as np
from client_utils import print_statistics
from urllib.request import urlretrieve
from pathlib import Path
import os

parser = argparse.ArgumentParser(description='REST Client for clip example')

parser.add_argument('--input_labels', required=False, default="cat,dog,wolf,tiger,man,horse,frog,tree,house,computer",
                    help="Specify input_labels to the CLIP model. default:cat,dog,wolf,tiger,man,horse,frog,tree,house,computer")
parser.add_argument('--image_url', required=False, default='https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg',
                    help='Specify image_url to send to the CLIP model. default:https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg')
parser.add_argument('--iterations', default=1,
                        help='Number of requests iterations, as default use number of images in numpy memmap. default: 1 ',
                        dest='iterations', type=int)

parser.add_argument('--url', required=False, default='localhost:8000',
                    help='Specify url to grpc service. default:localhost:8000')

args = vars(parser.parse_args())

iterations = args.get('iterations')
iteration = 0

url = args['url']
ssl_options = None

triton_client = httpclient.InferenceServerClient(
                url=url,
                ssl=False,
                ssl_options=ssl_options,
                verbose=False)

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

input_labels_array = [args['input_labels']]
input_labels = args['input_labels'].split(",")
print(f"Using input_labels:\n{input_labels}\n")

image_data = []
with open(sample_path, "rb") as f:
    image_data.append(f.read())

npydata = np.array(image_data, dtype=np.object_)
npylabelsdata = np.array(input_labels_array, dtype=np.object_)

inputs = []
inputs.append(httpclient.InferInput('image', [len(npydata)], "BYTES"))
inputs[0].set_data_from_numpy(npydata, binary_data=True)

inputs.append(httpclient.InferInput('input_labels', [len(npylabelsdata)], "BYTES"))
inputs[1].set_data_from_numpy(npylabelsdata, binary_data=True)

processing_times = []
for iteration in range(iterations):
    outputs = []
    print(f"Iteration {iteration}")
    start_time = datetime.datetime.now()

    model_name = "python_model"

    results = triton_client.infer(
                model_name=model_name,
                inputs=inputs)
    
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    processing_times.append(int(duration))

    print(f"Detection:\n{results.as_numpy('output_label').tobytes().decode()}\n")

print_statistics(np.array(processing_times,int), batch_size = 1)