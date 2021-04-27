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

import requests
import numpy as np
import base64
from tensorflow import make_tensor_proto, make_ndarray, make_tensor_proto
import datetime
import argparse
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

parser = argparse.ArgumentParser(description='Do requests to ie_serving and tf_serving using images in binary format')
parser.add_argument('--images_list', required=False, default='input_images.txt', help='path to a file with a list of labeled images')
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--input_name',required=False, default='input', help='Specify input tensor name. default: input')
parser.add_argument('--output_name',required=False, default='resnet_v1_50/predictions/Reshape_1', help='Specify output name. default: resnet_v1_50/predictions/Reshape_1')
parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. default: resnet',
                    dest='model_name')
parser.add_argument('--batchsize', default=1, help='Number of images in a single request. default: 1',
                    dest='batchsize')
args = vars(parser.parse_args())

address = "{}:{}/v1/models/{}:predict".format(args['rest_address'], args['rest_port'], args['model_name'])
input_images = args.get('images_list')
size = args.get('size')
with open(input_images) as f:
    lines = f.readlines()
batch_size = min(int(args.get('batchsize')), len(lines))
print('Start processing:')
print('\tModel name: {}'.format(args.get('model_name')))
print('\tImages list file: {}'.format(args.get('images_list')))

i = 0
matched = 0
processing_times = np.zeros((0),int)

batch_i = 0
image_data = []
labels = []
for line in lines:
    batch_i += 1
    path, label = line.strip().split(" ")
    with open(path, 'rb') as f:
        image_data.append(f.read())
    labels.append(label)
    if batch_i < batch_size:
        continue
    # Compose a JSON Predict request (send JPEG image in base64).
    jpeg_bytes = base64.b64encode(image_data).decode('utf-8')
    predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
    start_time = datetime.datetime.now()
    result = requests.post(address, data=predict_request)
    end_time = datetime.datetime.now()
    if args['output_name'] not in result.outputs:
        print("Invalid output name", args['output_name'])
        print("Available outputs:")
        for Y in result.outputs:
            print(Y)
        exit(1)
    duration = (end_time - start_time).total_seconds() * 1000
    processing_times = np.append(processing_times,np.array([int(duration)]))
    output = make_ndarray(result.outputs[args['output_name']])
    nu = np.array(output)
    # for object classification models show imagenet class
    print('Processing time: {:.2f} ms; speed {:.2f} fps'.format(round(duration, 2), round(1000 / duration, 2)))
    for i in range(nu.shape[0]):
        ma = np.argmax(nu[i])
        mark_message = ""
        if int(labels[i]) == ma:
            matched += 1
            mark_message = "; Correct match."
        else:
            mark_message = "; Incorrect match. Should be {} {}".format(label, classes.imagenet_classes[int(label)])
    i += 1
    print("\t", i, classes.imagenet_classes[ma], ma, mark_message)
    image_data = []
    labels = []

latency = np.average(processing_times / batch_size)
accuracy = matched / i

print("Overall accuracy=",accuracy*100,"%")
print("Average latency=",latency,"ms")