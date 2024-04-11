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
from re import I
import sys

import numpy as np
import datetime
import argparse

import tritonclient.http as httpclient


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sends requests via KServe REST API using binary encoded images.')
    parser.add_argument('--images_list', required=False, default='input_images.txt', help='path to a file with a list of images')
    parser.add_argument('--http_address',required=False, default='localhost',  help='Specify url to http service. default:localhost')
    parser.add_argument('--http_port',required=False, default=8000, help='Specify port to http service. default: 8000')
    parser.add_argument('--input_name',required=False, default='image', help='Specify input tensor name. default: image')
    parser.add_argument('--output_name',required=False, default='label',
                        help='Specify output name. default: label')
    parser.add_argument('--model_name', default='mobile_net', help='Define model name, must be same as in the service. default: mobile_net',

                        dest='model_name')

    args = vars(parser.parse_args())

    address = "{}:{}".format(args['http_address'],args['http_port'])
    input_name = args['input_name']
    output_name = args['output_name']

    ssl_options = None    

    processing_times = np.zeros((0),int)

    input_images = args.get('images_list')
    with open(input_images) as f:
        lines = f.readlines()

    print('Start processing:')
    print('\tModel name: {}'.format(args.get('model_name')))

    iteration = 0
    model_name = args.get('model_name')

    try:
        triton_client = httpclient.InferenceServerClient(
            url=address,
            ssl=False,
            ssl_options=ssl_options,
            verbose=False)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    processing_times = np.zeros((0),int)

    total_executed = 0
    matched_count = 0
    batch_i = 0
    image_data = []
    image_binary_size = []
    for line in lines:
        inputs = []
        batch_i += 1
        path = line.strip()
        with open(path, 'rb') as f:
            image_data.append(f.read())
        inputs.append(httpclient.InferInput(args['input_name'], [batch_i], "BYTES"))
        outputs = []
        outputs.append(httpclient.InferRequestedOutput(output_name, binary_data=True))
        
        inputs[0].set_data_from_numpy(np.array(image_data , dtype=np.object_))
        start_time = datetime.datetime.now()
        results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        processing_times = np.append(processing_times,np.array([int(duration)]))
        output = results.as_numpy(output_name)
        nu = np.array(output)
        print('{} classified  as {}'.format(path, nu[0].decode("utf-8")))
        print('Iteration {}; Processing time: {:.2f} ms; speed {:.2f} fps\n'.format(iteration,round(np.average(duration), 2),
                                                                                      round(1000  / np.average(duration), 2)
                                                                                      ))
        image_data = []
        batch_i = 0
    