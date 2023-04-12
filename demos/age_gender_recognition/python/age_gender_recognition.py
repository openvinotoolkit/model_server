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

import cv2
import numpy as np
import json
import requests
import argparse

parser = argparse.ArgumentParser(description='Client for age gender recognition')
parser.add_argument('--rest_address', required=False, default='localhost',  help='Specify url to REST API service. default:localhost')
parser.add_argument('--rest_port', required=False, default=9001, help='Specify port to REST API service. default: 9178')
parser.add_argument('--model_name', required=False, default='age_gender', help='Model name to request. default: age_gender')
parser.add_argument('--image_input_path', required=True, help='Input image path.')
parser.add_argument('--image_width', required=False, default=62, help='Pipeline input image width. default: 62')
parser.add_argument('--image_height', required=False, default=62, help='Pipeline input image height. default: 62')

args = vars(parser.parse_args())

def getJpeg(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # retrieved array has BGR format and 0-255 normalization
    img = cv2.resize(img, (args["image_height"], args["image_width"]))
    img = img.transpose(2,0,1).reshape(1, 3, args["image_height"], args["image_width"])
    print(path, img.shape, "; data range:",np.amin(img),":",np.amax(img))
    return img

my_image = getJpeg(args["image_input_path"])


data_obj = {'inputs':  my_image.tolist()}
data_json = json.dumps(data_obj)

result = requests.post(f'http://{args["rest_address"]}:{args["rest_port"]}/v1/models/{args["model_name"]}:predict', data=data_json, timeout=15)
result_dict = json.loads(result.text)
print(result_dict)
