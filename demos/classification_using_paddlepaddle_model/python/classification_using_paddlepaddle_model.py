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
import os, os.path
import sys
import json
import urllib.request
import cv2
import numpy as np
import argparse

from ovmsclient import make_grpc_client


def image_preprocess_mobilenetv3(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = np.transpose(img, [2,0,1]) / 255
    img = np.expand_dims(img, 0)
    img_mean = np.array([0.485, 0.456,0.406]).reshape((3,1,1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
    img -= img_mean
    img /= img_std
    return img.astype(np.float32)


def top_k(result, topk=5):
    imagenet_classes = json.loads(open("utils/imagenet_class_index.json").read())
    indices = np.argsort(-result[0])
    for i in range(topk):
        print(f"probability: {result[0][indices[i]]:.2f} => {imagenet_classes[str(indices[i])][1]}")


def build_parser():
    parser = argparse.ArgumentParser(description='Client for OCR pipeline')
    parser.add_argument('--grpc_address', required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
    parser.add_argument('--grpc_port', required=False, default=9000, help='Specify port to grpc service. default: 9000')
    parser.add_argument('--image_input_path', required=True, help='Image input path')
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    modelname = 'mobilenet'
    test_image = image_preprocess_mobilenetv3(args.image_input_path) 

    client = make_grpc_client(f"{args.grpc_address}:{args.grpc_port}")
    input_key = next(iter(client.get_model_metadata(model_name=modelname)['inputs']))

    classification_output = client.predict({ input_key: test_image }, modelname)

    #filter and print the top 5 results 
    top_k(classification_output)
