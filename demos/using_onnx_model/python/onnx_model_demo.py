#
# Copyright (c) 2022 Intel Corporation
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

import argparse
import numpy as np
from PIL import Image
import tritonclient.grpc as grpcclient
import classes


def load_image(path):
    with Image.open(path) as img:
        img = img.resize((224, 224))
        np_img = np.array(img, dtype=np.uint8)
        np_img = np_img[:, :, ::-1]
        np_img = np.expand_dims(np_img, axis=0)
    return np_img

parser = argparse.ArgumentParser(description='Run prediction on ONNX ResNet50 Model')
parser.add_argument('--image_path', required=False, default='../../common/static/images/bee.jpeg', help='Path to a file with a JPEG image')
parser.add_argument('--service_url',required=False, default='localhost:9001',  help='Specify url to grpc service. default:localhost:9001')
parser.add_argument('--output_name', required=False, default='gpu_0/softmax_1', help='Output tensor name. default: gpu_0/softmax_1')
args = vars(parser.parse_args())

input_name = "gpu_0/data_0"

print(f"Running inference with image: {args['image_path']}")
img = load_image(args["image_path"])

client = grpcclient.InferenceServerClient(url=args["service_url"])
infer_input = grpcclient.InferInput(input_name, img.shape, "UINT8")
infer_input.set_data_from_numpy(img)
result = client.infer("resnet", [infer_input])
output = result.as_numpy(args['output_name'])

max_idx = np.argmax(output)
print(f"Class with highest score: {max_idx}")
print(f"Detected class name: {classes.imagenet_classes[max_idx]}")
