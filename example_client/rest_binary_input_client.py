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

import base64
import requests
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Sends requests via TFS REST API using images in binary format.'
                                             'It displays performance statistics.')
parser.add_argument('--image_path', required=False, help='Image in binary format. default: https://tensorflow.org/images/blogs/serving/cat.jpg')
parser.add_argument('--rest_address',required=False, default='http://localhost',  help='Specify url to rest service. default: http://localhost')
parser.add_argument('--rest_port',required=False, default=9000, help='Specify port to rest service. default: 9000')
parser.add_argument('--input_name',required=False, default='input', help='Specify input tensor name. default: input')
parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. default: resnet',
                    dest='model_name')
parser.add_argument('--iterations', default=10, help='Number of requests iterations. default: 10', dest='iterations', type=int)

args = vars(parser.parse_args())
# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

if args['image_path']:
  with open(args['image_path'], 'rb') as f:
    data = f.read()
else:
  # Download the image since we weren't given one
  dl_request = requests.get(IMAGE_URL, stream=True)
  dl_request.raise_for_status()
  data = dl_request.content

address = "{}:{}/v1/models/{}:predict".format(args['rest_address'], args['rest_port'], args['model_name'])
# Compose a JSON Predict request (send JPEG image in base64).
jpeg_bytes = base64.b64encode(data).decode('utf-8')
predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes

# Send few requests to warm-up the model.
for _ in range(3):
  response = requests.post(address, data=predict_request)
  response.raise_for_status()

# Send few actual requests and report average latency.
total_time = 0
for _ in range(args['iterations']):
  response = requests.post(address, data=predict_request)
  response.raise_for_status()
  total_time += response.elapsed.total_seconds()
  prediction = response.json()['predictions'][0]

print('Prediction class: {}, avg latency: {} ms'.format(
      prediction['classes'], (total_time*1000)/num_requests))
