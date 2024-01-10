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
import tritonclient.grpc as grpcclient
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Client for clip example')

parser.add_argument('--url', required=False, default='localhost:9000',
                    help='Specify url to grpc service. default:localhost:9000')
parser.add_argument('--input_labels', required=False, default="cat,dog,wolf,tiger,man,horse,frog,tree,house,computer",
                    help="Specify input_labels to the CLIP model. default:cat,dog,wolf,tiger,man,horse,frog,tree,house,computer")
parser.add_argument('--image_url', required=False, default='https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg',
                    help='Specify image_url to send to the CLIP model. default:https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg')
args = vars(parser.parse_args())

client = grpcclient.InferenceServerClient(args['url'])
image_url = args['image_url']
print(f"Using image_url:\n{image_url}\n")
data = image_url.encode()
infer_input = grpcclient.InferInput("image_url", [len(data)], "BYTES")
infer_input._raw_content = data
input_labels = args['input_labels'].split(",")

labels_npy = np.array(input_labels)
infer_input1 = grpcclient.InferInput("input_labels", [len(labels_npy)], labels_npy.dtype.str)
infer_input1._raw_content = labels_npy.tobytes()

results = client.infer("python_model", [infer_input, infer_input1])
print(f"detection:\n{results.as_numpy('detection')}\n")
