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
from re import I
import sys
sys.path.append("../../../../demos/common/python")

import numpy as np
import classes
import datetime
import argparse
from client_utils import print_statistics
import requests
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sends requests via KServe REST API using binary encoded images. '
                                                 'It displays performance statistics and optionally the model accuracy')
    parser.add_argument('--images_list', required=False, default='input_images.txt', help='path to a file with a list of labeled images')
    parser.add_argument('--labels_numpy_path', required=False, help='numpy in shape [n,1] - can be used to check model accuracy')
    parser.add_argument('--http_address',required=False, default='localhost',  help='Specify url to http service. default:localhost')
    parser.add_argument('--http_port',required=False, default=5000, help='Specify port to http service. default: 5000')
    parser.add_argument('--input_name',required=False, default='input', help='Specify input tensor name. default: input')
    parser.add_argument('--output_name',required=False, default='resnet_v1_50/predictions/Reshape_1',
                        help='Specify output name. default: resnet_v1_50/predictions/Reshape_1')
    parser.add_argument('--batchsize', default=1,
                        help='Number of images in a single request. default: 1',
                        dest='batchsize')
    parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. default: resnet',
                        dest='model_name')
    parser.add_argument('--pipeline_name', default='', help='Define pipeline name, must be same as is in service',
                        dest='pipeline_name')

    error = False
    args = vars(parser.parse_args())

    address = "{}:{}".format(args['http_address'],args['http_port'])
    input_name = args['input_name']
    output_name = args['output_name']

    processing_times = np.zeros((0),int)

    input_images = args.get('images_list')
    with open(input_images) as f:
        lines = f.readlines()
    batch_size = int(args.get('batchsize'))
    while batch_size > len(lines):
        lines += lines

    if args.get('labels_numpy_path') is not None:
        lbs = np.load(args['labels_numpy_path'], mmap_mode='r', allow_pickle=False)
        matched_count = 0
        total_executed = 0
    batch_size = int(args.get('batchsize'))

    print('Start processing:')
    print('\tModel name: {}'.format(args.get('pipeline_name') if bool(args.get('pipeline_name')) else args.get('model_name')))

    iteration = 0
    is_pipeline_request = bool(args.get('pipeline_name'))

    model_name = args.get('pipeline_name') if is_pipeline_request else args.get('model_name')

    url = f"http://{address}/v2/models/{model_name}/infer"
    http_session = requests.session()

    batch_i = 0
    image_data = []
    image_binary_size = []
    labels = []
    for line in lines:
        batch_i += 1
        path, label = line.strip().split(" ")
        with open(path, 'rb') as f:
            image_data.append(f.read())
            image_binary_size.append(len(image_data[-1]))
        labels.append(label)
        if batch_i < batch_size:
            continue
        image_binary_size_str = ",".join(map(str, image_binary_size))
        inference_header = {"inputs":[{"name":input_name,"shape":[batch_i],"datatype":"BYTES","parameters":{"binary_data_size":image_binary_size_str}}]}
        inference_header_binary = json.dumps(inference_header).encode()

        start_time = datetime.datetime.now()
        results = http_session.post(url, inference_header_binary + b''.join(image_data), headers={"Inference-Header-Content-Length":str(len(inference_header_binary))})
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        processing_times = np.append(processing_times,np.array([int(duration)]))
        results_dict = json.loads(results.text)
        if "error" in results_dict.keys():
            print(f"Error: {results_dict['error']}")
            error = True
            break
            
        output = np.array(json.loads(results.text)['outputs'][0]['data'])
        output_shape = tuple(results_dict['outputs'][0]['shape'])
        nu = np.reshape(output, output_shape)
        # for object classification models show imagenet class
        print('Iteration {}; Processing time: {:.2f} ms; speed {:.2f} fps'.format(iteration,round(np.average(duration), 2),
                                                                                    round(1000 * batch_size / np.average(duration), 2)
                                                                                    ))
        # Comment out this section for non imagenet datasets
        print("imagenet top results in a single batch:")
        for i in range(nu.shape[0]):
            lbs_i = iteration * batch_size
            single_result = nu[[i],...]
            offset = 0
            if nu.shape[1] == 1001:
                offset = 1
            ma = np.argmax(single_result) - offset
            mark_message = ""
            if args.get('labels_numpy_path') is not None:
                total_executed += 1
                if ma == lbs[lbs_i + i]:
                    matched_count += 1
                    mark_message = "; Correct match."
                else:
                    mark_message = "; Incorrect match. Should be {} {}".format(lbs[lbs_i + i], classes.imagenet_classes[lbs[lbs_i + i]] )
            print("\t",i, classes.imagenet_classes[ma],ma, mark_message)
        # Comment out this section for non imagenet datasets
        iteration += 1
        image_data = []
        image_binary_size = []
        labels = []
        batch_i = 0

    if not error:
        print_statistics(processing_times, batch_size)

        if args.get('labels_numpy_path') is not None:
            print('Classification accuracy: {:.2f}'.format(100*matched_count/total_executed))

