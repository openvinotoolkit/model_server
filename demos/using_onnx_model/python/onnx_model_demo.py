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
from ovmsclient import make_grpc_client
import classes


def load_image(path):
    with Image.open(path) as img:
        img = img.resize((224, 224))
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.expand_dims(np_img, axis=0)
    return np_img

parser = argparse.ArgumentParser(description='Run prediction on ONNX ResNet50 Model')
parser.add_argument('--image_path', required=False, default='../../common/static/images/bee.jpeg', help='Path to a file with a JPEG image')
parser.add_argument('--service_url',required=False, default='localhost:9001',  help='Specify url to grpc service. default:localhost:9001')
parser.add_argument('--load_image', action="store_true", required=False, help='Send image after loading it with Pillow')
args = vars(parser.parse_args())

print(f"Running inference with image: {args['image_path']}")
if args["load_image"]:
    img = load_image(args["image_path"])
else:
    with open(args["image_path"], "rb") as f:
        img = f.read()

input_name = "gpu_0/data_0"

client = make_grpc_client(args["service_url"])
output = client.predict({input_name: img}, "resnet")

max = np.argmax(output)
print(f"Class with highest score: {max}")
print(f"Detected class name: {classes.imagenet_classes[max]}")
