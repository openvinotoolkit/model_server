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

def convert_pose():
    print("Converting pose detection model")
    run_command("wget https://github.com/PINTO0309/tflite2tensorflow/raw/main/schema/schema.fbs")
    run_command("git clone -b v2.0.8 https://github.com/google/flatbuffers.git")
    run_command("bash build_dependencies.sh")
    run_command("cp -r  ovms/pose_detection/1/pose_detection.tflite .")
    run_command("tflite2tensorflow --model_path pose_detection.tflite --flatc_path flatbuffers/build/flatc --schema_path schema.fbs --output_pb")
    run_command("tflite2tensorflow --model_path pose_detection.tflite --flatc_path flatbuffers/build/flatc --schema_path schema.fbs --output_no_quant_float32_tflite   --output_dynamic_range_quant_tflite   --output_weight_quant_tflite   --output_float16_quant_tflite   --output_integer_quant_tflite")
    run_command("cp -rf saved_model/model_float32.tflite ovms/pose_detection/1/pose_detection.tflite")
    run_command("rm -rf pose_detection.tflite")
    run_command("rm -rf saved_model")

def download_model(model_path: str):
    """Downloads the oss model from Google Cloud Storage if it doesn't exist in the package."""
    model_url = _GCS_URL_PREFIX + model_path.split('/')[-1]
    dst = os.path.join("ovms/", model_path.replace("/","/1/"))
    dst_dir = os.path.dirname(model_path)

    # Workaround to copy every model in separate directory
    model_name = os.path.basename(model_path).replace(".tflite","")
    dir_name = os.path.basename(dst_dir)
    if dir_name != model_name:
        dst = dst.replace(dir_name + "/", model_name + "/")

    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    dst_file = os.path.join(dst_dir, os.path.basename(model_path))
    print('Downloading model to ' + dst_file)
    with urllib.request.urlopen(model_url) as response, open(dst_file,
                                                           'wb') as out_file:
        if response.code != 200:
            raise ConnectionError('Cannot download ' + model_path +
                                    ' from Google Cloud Storage.')
        shutil.copyfileobj(response, out_file)

def prepare_models():    
    external_files = [
       # Using short range
       # 'face_detection/face_detection_full_range_sparse.tflite',
        'face_detection/face_detection_short_range.tflite',
        'face_landmark/face_landmark.tflite',
       # Model loading error due to custom tf op
       # 'face_landmark/face_landmark_with_attention.tflite',
        'hand_landmark/hand_landmark_full.tflite',
       # Using full
       # 'hand_landmark/hand_landmark_lite.tflite',
        'holistic_landmark/hand_recrop.tflite',
        'iris_landmark/iris_landmark.tflite',
        'palm_detection/palm_detection_full.tflite',
       # Using full
       # 'palm_detection/palm_detection_lite.tflite',
       # Need to convert model using TF toolset
        'pose_detection/pose_detection.tflite',
        'pose_landmark/pose_landmark_full.tflite',
    ]
    for elem in external_files:
      sys.stderr.write('downloading file: %s\n' % elem)
      download_model(elem)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sends requests via KServe gRPC API using images in format supported by OpenCV. '
                                                 'It displays performance statistics and optionally the model accuracy')
    parser.add_argument('--images_list', required=False, default='input_images.txt', help='path to a file with a list of labeled images')
    parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
    parser.add_argument('--grpc_port',required=False, default=9022, help='Specify port to grpc service. default: 9022')
    parser.add_argument('--input_name',required=False, default='first_input_video', help='Specify input tensor name. default: input')
    parser.add_argument('--output_name',required=False, default='output',
                        help='Specify output name. default: output')
    parser.add_argument('--batchsize', default=1,
                        help='Number of images in a single request. default: 1',
                        dest='batchsize')
    parser.add_argument('--graph_name', default='holisticTracking', help='Define model name, must be same as is in service. default: holisticTracking',
                        dest='graph_name')
    parser.add_argument('--tls', default=False, action='store_true', help='use TLS communication with GRPC endpoint')

    parser.add_argument('--download_models', default=False, action='store_true', help='download models and files for demo')

    error = False
    args = vars(parser.parse_args())

    if args['download_models'] == True:
        # TODO uncomment when download is available
        #run_command("curl -kL -o girl.jpeg https://cdn.pixabay.com/photo/2019/03/12/20/39/girl-4051811_960_720.jpg")
        run_command("mkdir -p mediapipe/mediapipe/modules/hand_landmark/")
        run_command("mkdir -p ovms")
        run_command("wget -O mediapipe/mediapipe/modules/hand_landmark/handedness.txt https://raw.githubusercontent.com/openvinotoolkit/mediapipe/main/mediapipe/modules/hand_landmark/handedness.txt")
        run_command("cp config_holistic.json ovms/")
        run_command("cp holistic_tracking.pbtxt ovms/")
        prepare_models()
        convert_pose()
        print("All files and models are prepared.")
        exit(0)

    print("Running demo application.")
    address = "{}:{}".format(args['grpc_address'],args['grpc_port'])
    input_name = args['input_name']
    output_name = args['output_name']

    processing_times = np.zeros((0),int)

    input_images = args.get('images_list')
    with open(input_images) as f:
        lines = f.readlines()
    batch_size = int(args.get('batchsize'))
    while batch_size > len(lines):
        lines += lines

    batch_size = int(args.get('batchsize'))

    print('Start processing:')
    print('\tModel name: {}'.format(args.get('graph_name')))

    iteration = 0
    is_pipeline_request = bool(args.get('pipeline_name'))

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

    for line in lines:
        inputs = []
        if not os.path.exists(line.strip()):
            print("Image does not exist: " + line.strip())
        im_cv = cv2.imread(line.strip()) 
        img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        print(img.shape)
        inputs.append(grpcclient.InferInput(args['input_name'], img.shape, "UINT8"))
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput(output_name))
        
        inputs[0].set_data_from_numpy(img)
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
        out = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite("image_" + str(iteration) + ".jpg", out)
        iteration = iteration + 1
