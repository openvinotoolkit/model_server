#
# Copyright (c) 2023 Intel Corporation
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
import argparse
import datetime
import numpy as np
from client_utils import print_statistics

parser = argparse.ArgumentParser(description='Client for clip example')

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

client = grpcclient.InferenceServerClient(args['url'])
image_url = args['image_url']
print(f"Using image_url:\n{image_url}\n")
data = image_url.encode()
infer_input = grpcclient.InferInput("image_url", [len(data)], "BYTES")
infer_input._raw_content = data
input_labels = args['input_labels'].split(",")
print(f"Using input_labels:\n{input_labels}\n")

labels_npy = np.array(input_labels)
infer_input1 = grpcclient.InferInput("input_labels", [len(labels_npy)], labels_npy.dtype.str)
infer_input1._raw_content = labels_npy.tobytes()

processing_times = np.zeros((0),int)
while iteration < iterations:
    iteration += 1
    print(f"Iteration {iteration}")
    start_time = datetime.datetime.now()
    results = client.infer("python_model", [infer_input, infer_input1])
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    processing_times = np.append(processing_times,np.array([int(duration)]))
    print(f"Detection:\n{results.as_numpy('detection').tobytes().decode()}\n")

batch_size = 1
print_statistics(processing_times, batch_size)