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
import sys
sys.path.append("../../../../demos/common/python")

import requests
import numpy as np
import base64
import json
import classes
import datetime
import argparse

def create_request(image_data, request_format):
    signature = "serving_default"
    instances = []
    if request_format == "row_name":
        for image in image_data:
            jpeg_bytes = base64.b64encode(image).decode('utf-8')
            instances.append({args['input_name']: {"b64": jpeg_bytes}})
    else:
        for image in image_data:
            jpeg_bytes = base64.b64encode(image).decode('utf-8')
            instances.append({"b64": jpeg_bytes})
    if request_format == "row_name":
        data_obj = {"signature_name": signature, "instances": instances}
    elif request_format == "row_noname":
        data_obj = {"signature_name": signature, "instances": instances}
    elif request_format == "column_name":
        data_obj = {"signature_name": signature, 'inputs': {args['input_name']: instances}}
    elif request_format == "column_noname":
        data_obj = {"signature_name": signature, 'inputs': instances}
    else:
        print("invalid request format defined")
        exit(1)
    data_json = json.dumps(data_obj)
    return data_json

parser = argparse.ArgumentParser(description='Sends requests via TensorFlow Serving RESTful API using images in binary format. '
                                             'It displays performance statistics and optionally the model accuracy')
parser.add_argument('--images_list', required=False, default='input_images.txt',
                    help='path to a file with a list of labeled images')
parser.add_argument('--rest_url', required=False, default='http://localhost:8000',
                    help='Specify url to REST API service. default: http://localhost:8000')
parser.add_argument('--input_name', required=False, default='image_bytes',
                    help='Specify input tensor name. default: image_bytes')
parser.add_argument('--output_name', required=False, default='probabilities',
                    help='Specify output name. default: probabilities')
parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. default: resnet',
                    dest='model_name')
parser.add_argument('--request_format', default='row_noname', help='Request format according to TF Serving API: row_noname,row_name,column_noname,column_name',
                    choices=["row_noname", "row_name", "column_noname", "column_name"], dest='request_format')
# If input numpy file has too few frames according to the value of iterations and the batch size, it will be
# duplicated to match requested number of frames
parser.add_argument('--batchsize', default=1, help='Number of images in a single request. default: 1',
                    dest='batchsize')
args = vars(parser.parse_args())

address = "{}/v1/models/{}:predict".format(
    args['rest_url'], args['model_name'])
input_images = args.get('images_list')
with open(input_images) as f:
    lines = f.readlines()
batch_size = int(args.get('batchsize'))
while batch_size > len(lines):
    lines += lines
print('Start processing:')
print('\tModel name: {}'.format(args.get('model_name')))
print('\tImages list file: {}'.format(args.get('images_list')))

count = 0
matched = 0
processing_times = np.zeros((0), int)

batch_i = 0
image_data = []
labels = []
session = requests.Session()
for line in lines:
    batch_i += 1
    path, label = line.strip().split(" ")
    with open(path, 'rb') as f:
        image_data.append(f.read())
    labels.append(label)
    if batch_i < batch_size:
        continue
    # Compose a JSON Predict request (send JPEG image in base64).
    predict_request = create_request(image_data, args['request_format'])
    start_time = datetime.datetime.now()
    result = session.post(address, data=predict_request)
    end_time = datetime.datetime.now()
    try:
        result_dict = json.loads(result.text)
    except ValueError:
        print("The server response is not json format: {}",format(result.text))
        exit(1)
    if "error" in result_dict:
        print('Server returned error: {}'.format(result_dict))
        exit(1)

    if "outputs" in result_dict:  # is column format
        keyname = "outputs"
        if type(result_dict[keyname]) is dict:
            if args['output_name'] not in result_dict[keyname]:
                print("Invalid output name", args['output_name'])
                print("Available outputs:")
                for Y in result_dict[keyname]:
                    print(Y)
                exit(1)
            output = result_dict[keyname][args['output_name']]
        else:
            output = result_dict[keyname]
    elif "predictions" in result_dict:  # is row format
        keyname = "predictions"
        if type(result_dict[keyname][0]) is dict:  # are multiple outputs
            output = []
            for row in result_dict[keyname]:  # iterate over all results in the batch
                output.append(row[args['output_name']])
        else:
            output = result_dict[keyname]
    else:
        print("Missing required response in {}".format(result_dict))
        exit(1)
    duration = (end_time - start_time).total_seconds() * 1000
    processing_times = np.append(processing_times, np.array([int(duration)]))
    # for object classification models show imagenet class
    print('Batch: {}; Processing time: {:.2f} ms; speed {:.2f} fps'.format(
        count // batch_size, round(duration, 2), round(1000 / duration, 2)))

    nu = np.array(output)  # numpy array with inference results
    print("output shape: {}".format(nu.shape))
    for i in range(nu.shape[0]):
        single_result = nu[[i], ...]
        offset = 0
        if nu.shape[1] == 1001:
            offset = 1 
        ma = np.argmax(single_result) - offset
        mark_message = ""
        if int(labels[i]) == ma:
            matched += 1
            mark_message = "; Correct match."
        else:
            mark_message = "; Incorrect match. Should be {} {}".format(
                label, classes.imagenet_classes[int(label)])
        count += 1
        print("\t", count, classes.imagenet_classes[ma], ma, mark_message)
    image_data = []
    labels = []
    batch_i = 0

latency = np.average(processing_times)
accuracy = matched / count

print("Overall accuracy=", accuracy*100, "%")
print("Average latency=", latency, "ms")
