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
    parser.add_argument('--input_video', required=False, default='test_video.mp4', help='path to a file input video')
    parser.add_argument('--output_video', required=False, default='output_video.mp4', help='path to a file output video')
    parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
    parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
    parser.add_argument('--input_name',required=False, default='input_video', help='Specify input name. default: input')
    parser.add_argument('--output_name',required=False, default='output_video',
                        help='Specify output name. default: output')
    parser.add_argument('--graph_name', default='objectTracking', help='Define model name, must be same as is in service. default: objectDetection',
                        dest='graph_name')
    parser.add_argument('--tls', default=False, action='store_true', help='use TLS communication with GRPC endpoint')
    parser.add_argument('--download_models', default=False, action='store_true', help='download models and files for demo')

    error = False
    args = vars(parser.parse_args())

    if args['download_models'] == True:
        download_model('models/ssdlite_object_detection.tflite')
        download_model('ssdlite_object_detection_labelmap.txt')
        exit(0)
    address = "{}:{}".format(args['grpc_address'], args['grpc_port'])
    input_name = args['input_name']
    output_name = args['output_name']
    graph_name = args['graph_name']

    # --- Connect to OVMS ---
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=address,
            ssl=args['tls'],
            verbose=False)
    except Exception as e:
        print("Context creation failed: " + str(e))
        sys.exit()

    # --- Open input video ---
    cap = cv2.VideoCapture(args['input_video'])
    if not cap.isOpened():
        print("Error: Cannot open video file: " + args['input_video'])
        sys.exit(-1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")

    # --- Prepare output video writer ---
    os.makedirs("./results", exist_ok=True)
    output_path = os.path.join("./results", args['output_video'])
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_writer = None  # initialize after first frame so we know output shape

    processing_times = np.zeros((0), int)
    iteration = 0

    print(f"Start processing graph: {graph_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Build gRPC request --- 
        # KEY DIFFERENCE from object_detection:
        # We send frames one by one but as a video stream input
        # The graph maintains tracking state across frames server-side
        frame_timestamp_us = int((iteration / fps) * 1e6)

        inputs = []
        outputs = []

        # inputs.append(grpcclient.InferInput(input_name, frame.shape, "UINT8"))
        # nmpy = np.array(frame, dtype=np.uint8)
        # inputs[0].set_data_from_numpy(nmpy)
        inputs = [grpcclient.InferInput(input_name, frame.shape, "UINT8")]
        nmpy = np.array(frame, dtype=np.uint8)
        inputs[0].set_data_from_numpy(nmpy)

        outputs.append(grpcclient.InferRequestedOutput(output_name))

        start_time = datetime.datetime.now()
        try:
            results = triton_client.infer(
                model_name=graph_name,
                inputs=inputs,
                outputs=outputs,
                headers={"mediapipe-packet-timestamp": str(frame_timestamp_us)}
            )
        except Exception as e:
            print(f"Frame {iteration} inference failed: {str(e)}")
            break

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        processing_times = np.append(processing_times, np.array([int(duration)]))

        # --- Get output frame ---
        output_frame = results.as_numpy(output_name)
        nu = np.array(output_frame, dtype=np.uint8)

        # Initialize writer with actual output frame dimensions
        if out_writer is None:
            out_h, out_w = nu.shape[:2]
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
            print(f"Output video: {out_w}x{out_h}")

        out_writer.write(nu)

        print('Frame {}; Processing time: {:.2f} ms; speed {:.2f} fps'.format(
            iteration,
            round(np.average(duration), 2),
            round(1000 / np.average(duration), 2)
        ))

        iteration += 1

    # --- Cleanup ---
    cap.release()
    if out_writer:
        out_writer.release()

    print(f"\nProcessed {iteration} frames")
    print(f"Average latency: {round(np.average(processing_times), 2)} ms")
    print(f"Average speed: {round(1000 / np.average(processing_times), 2)} fps")
    print(f"Output saved to: {output_path}")