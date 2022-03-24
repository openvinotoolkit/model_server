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
import cv2
from ovmsclient import make_grpc_client
import classes

def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

def getJpeg(path, size):
    with open(path, mode='rb') as file:
        content = file.read()

    img = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # BGR format
    # format of data is HWC
    # add image preprocessing if needed by the model
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32')
    #convert to NCHW
    img = img.transpose(2,0,1)
    # normalize to adjust to model training dataset
    img = preprocess(img)
    img = img.reshape(1,3,size,size)
    print(path, img.shape, "; data range:",np.amin(img),":",np.amax(img))
    return img


parser = argparse.ArgumentParser(description='Run prediction on ONNX ResNet50 Model')
parser.add_argument('--image_path', required=False, default='../../common/static/images/bee.jpeg', help='Path to a file with a JPEG image')
parser.add_argument('--service_url',required=False, default='localhost:9001',  help='Specify url to grpc service. default:localhost:9001')
parser.add_argument('--run_preprocessing',required=False, action="store_true", help='Specify if preprocessing should be run on the client side. default: False')
args = vars(parser.parse_args())

if args["run_preprocessing"]:
    print("Running with preprocessing on client side")
    img = getJpeg(args["image_path"], 224)
    input_name = "gpu_0/data_0"
else:
    print("Running without preprocessing on client side")
    with open(args["image_path"], "rb") as f:
        img = f.read()
    input_name = "0"

client = make_grpc_client(args["service_url"])
output = client.predict({input_name: img}, "resnet")

max = np.argmax(output)
print("Class is with highest score: {}".format(max))
print("Detected class name: {}".format(classes.imagenet_classes[max]))
