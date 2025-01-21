#
# Copyright (c) 2023 Intel Corporation
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

import numpy as np
import cv2

import datetime
import argparse
import os
import subprocess
import shutil
import urllib.request

import tritonclient.grpc as grpcclient

_GCS_URL_PREFIX = 'https://storage.googleapis.com/mediapipe-assets/'

def run_command(command):
    print(command)
    if subprocess.call(command.split()) != 0:
      sys.exit(-1)

def download_model(model_path: str):
    """Downloads the oss model from Google Cloud Storage if it doesn't exist in the package."""
    model_url = _GCS_URL_PREFIX + model_path.split('/')[-1]
    dst = model_path.replace("/","/1/")
    dst_dir = os.path.dirname(model_path)

    # Workaround to copy every model in separate directory
    model_name = os.path.basename(model_path).replace(".tflite","")
    dir_name = os.path.basename(dst_dir)
    if dir_name != model_name:
        dst = dst.replace(dir_name + "/", model_name + "/")

    dst_dir = os.path.dirname(dst)
    if model_path == 'ssdlite_object_detection_labelmap.txt':
        dst_dir = ''
    elif not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    dst_file = os.path.join(dst_dir, os.path.basename(model_path))
    print('Downloading model to ' + dst_file)
    with urllib.request.urlopen(model_url) as response, open(dst_file,
                                                           'wb') as out_file:
        if response.code != 200:
            raise ConnectionError('Cannot download ' + model_path +
                                    ' from Google Cloud Storage.')
        shutil.copyfileobj(response, out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sends requests via KServe gRPC API using images in format supported by OpenCV. ')
    parser.add_argument('--images_list', required=False, default='input_images.txt', help='path to a file with a list of labeled images')
    parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
    parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
    parser.add_argument('--input_name',required=False, default='input', help='Specify input name. default: input')
    parser.add_argument('--output_name',required=False, default='output',
                        help='Specify output name. default: output')
    parser.add_argument('--graph_name', default='objectDetection', help='Define model name, must be same as is in service. default: objectDetection',
                        dest='graph_name')
    parser.add_argument('--tls', default=False, action='store_true', help='use TLS communication with GRPC endpoint')
    parser.add_argument('--download_models', default=False, action='store_true', help='download models and files for demo')

    error = False
    args = vars(parser.parse_args())

    if args['download_models'] == True:
        download_model('models/ssdlite_object_detection.tflite')
        download_model('ssdlite_object_detection_labelmap.txt')
        exit(0)

    address = "{}:{}".format(args['grpc_address'],args['grpc_port'])
    input_name = args['input_name']
    output_name = args['output_name']

    processing_times = np.zeros((0),int)

    input_images = args.get('images_list')
    with open(input_images) as f:
        lines = f.readlines()

    print('Start processing:')
    print('\tGraph name: {}'.format(args.get('graph_name')))

    iteration = 0

    graph_name = args.get('graph_name')

    try:
        triton_client = grpcclient.InferenceServerClient(
            url=address,
            ssl=args['tls'],
            verbose=False)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    processing_times = np.zeros((0),int)
    exist = os.path.exists("./results")
    if not exist:
        os.mkdir("./results")

    for line in lines:
        inputs = []
        path = line.strip()
        if not os.path.exists(path):
            print("Image does not exist: " + path)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) 
        print(os.path.basename(path))
        inputs.append(grpcclient.InferInput(args['input_name'], img.shape, "UINT8"))
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput(output_name))
        
        nmpy = np.array(img , dtype=np.uint8)
        inputs[0].set_data_from_numpy(nmpy)
        start_time = datetime.datetime.now()
        results = triton_client.infer(model_name=graph_name,
                                  inputs=inputs,
                                  outputs=outputs)
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        processing_times = np.append(processing_times,np.array([int(duration)]))
        output = results.as_numpy(output_name)
        nu = np.array(output)
        print('Iteration {}; Processing time: {:.2f} ms; speed {:.2f} fps'.format(iteration,round(np.average(duration), 2),
                                                                                      round(1000 / np.average(duration), 2)
                                                                                      ))
        cv2.imwrite("./results/received_" + os.path.basename(path), output)
        iteration = iteration + 1
