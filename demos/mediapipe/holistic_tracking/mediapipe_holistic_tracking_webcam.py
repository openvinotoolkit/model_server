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

import argparse
import subprocess

import tritonclient.grpc as grpcclient

def run_command(command):
    print(command)
    if subprocess.call(command.split()) != 0:
      sys.exit(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sends requests via KServe gRPC API using images in format supported by OpenCV. '
                                                 'It displays performance statistics and optionally the model accuracy')
    parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
    parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
    parser.add_argument('--input_device',required=False, default='0', help='Specify input device name for cv.VideoCapture() default: 0')
    parser.add_argument('--output',required=False, default=None,
                        help='Specify output file name. default: None')
    parser.add_argument('--tls', default=False, action='store_true', help='use TLS communication with GRPC endpoint')

    error = False
    args = vars(parser.parse_args())

    input_device  = args['input_device']
    if str(input_device).isdecimal():
        input_device = int(input_device)
    video_in = cv2.VideoCapture(input_device)
    if not video_in.isOpened():
        raise(f'Failed to open video input ({input_device})')
    
    ret, img = video_in.read()
    if args['output'] is not None:
        img_h, img_w, _ = img.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(args['output'], fourcc, 30.0, (img_w, img_h))
    else:
        video_out = None
    
    address = "{}:{}".format(args['grpc_address'],args['grpc_port'])

    try:
        triton_client = grpcclient.InferenceServerClient(
            url=address,
            ssl=args['tls'],
            verbose=False)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    key = -1
    while key != 27 and key != ord('q'):
        ret, im_cv = video_in.read()
        if ret == False:
            break
        img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        input = grpcclient.InferInput('first_input_video', img.shape, 'UINT8')
        output = grpcclient.InferRequestedOutput('output')
        input.set_data_from_numpy(img)
        results = triton_client.infer(model_name='holisticTracking',
                                  inputs=[input],
                                  outputs=[output])
        output = results.as_numpy('output')
        out = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        if video_out is not None:
            video_out.write(out)

        cv2.imshow('Result', out)
        key = cv2.waitKey(15)

    if video_out is not None:
        video_out.release()
    video_in.release()
    cv2.destroyAllWindows()

"""
docker run --rm -d \
    -v $PWD/mediapipe:/mediapipe \
    --name ovms -v $PWD/ovms:/models \
    -p 9000:9000 openvino/model_server:latest \
    --config_path /models/config_holistic.json \
    --port 9000
"""
