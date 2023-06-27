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

import cv2
import time
import argparse
from functools import partial
import tritonclient.grpc as grpcclient
import numpy as np
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--grpc_address', required=False, default='localhost:9000', help='Specify url to grpc service')
parser.add_argument('--input_stream', required=False, default="rtsp://localhost:8080/channel1", type=str, help='Url of input rtsp stream')
parser.add_argument('--output_stream', required=False, default="rtsp://localhost:8080/channel2", type=str, help='Url of output rtsp stream')
parser.add_argument('--model_name', required=False, default="detect_text_images", type=str, help='Name of the model')
parser.add_argument('--width', required=False, default=704, type=int, help='Width of model\'s input image')
parser.add_argument('--height', required=False, default=704, type=int, help='Height of model\'s input image')
parser.add_argument('--verbose', required=False, default=False, type=bool, help='Height of model\'s input image')
parser.add_argument('--input_name', required=False, default="image", type=str, help='Name of the model\'s input')
args = parser.parse_args()
output_stream = args.output_stream

triton_client = grpcclient.InferenceServerClient( url=args.grpc_address, verbose=False)

WIDTH = args.width
HEIGHT = args.height
cap = cv2.VideoCapture(args.input_stream, cv2.CAP_FFMPEG)
fps = cap.get(cv2.CAP_PROP_FPS)
force_exit = False

def decode(text):
    word = ''
    last_character = None
    for character in text:
        if character == last_character:
            continue
        elif character == '#':
            last_character = None
        else:
            last_character = character
            word += character
    return word

def get_text(output):
    alphabet = '#1234567890abcdefghijklmnopqrstuvwxyz'
    preds = output.argmax(2)
    word = ''
    for pred in preds:
        word += alphabet[pred[0]]
    return decode(word)

def finish():
    global force_exit
    force_exit = True

def grab_frame(cap):
    success, frame = cap.read()
    if not success:
        print("[WARNING] No Input frame")
        finish()
        return None

    if frame.shape[0] > HEIGHT and frame.shape[1] > WIDTH:
        frame = cv2.resize(frame, (HEIGHT, WIDTH), interpolation=cv2.INTER_LINEAR)
    else:
        frame = cv2.resize(frame, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)
    return frame

def postprocess(frame, result):
    #result: 
        #name: "texts"
        #datatype: "FP32"
        #shape: <number_of_detections> <output_sequence_length> 1 <confidence_distribution_across_alphanumeric_symbols>
 
        #name: "confidence_levels"
        #datatype: "FP32"
        #shape: <number_of_detections> 1 1

        #name: "text_coordinates"
        #datatype: "INT32"
        #shape: <number_of_detections> 1 4

        #name: "text_images"
        #datatype: "FP32"
        #shape: <number_of_detections> 1 32 128 1
    print(result.__dict__)
    if result is not None:
        texts = result.as_numpy("texts")
        text_coordinates = result.as_numpy('text_coordinates')
        for i in range(len(texts)):
            text = get_text(texts[i])
            coords = text_coordinates[i][0]
            x_min = coords[0]
            y_min = coords[1]
            x_max = coords[0] + coords[2]
            y_max = coords[1] + coords[3]
            frame = cv2.rectangle(frame, (x_min,y_min), (x_max,y_max), (0,0,255), 1)
            cv2.putText(frame, text, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            if args.verbose:
                print(text)
                print((x_min,y_min), (x_max,y_max))
    return frame


def callback(frame, result, error):
    frame = postprocess(frame, result)
    ffmpeg_process.stdin.write(frame.astype(np.uint8).tobytes())
    if error is not None:
        print(error)


def open_ffmpeg_stream_process():
    args = (
        "ffmpeg -re -stream_loop -1 -f rawvideo -pix_fmt "
        f"bgr24 -s {WIDTH}x{HEIGHT} -i pipe:0 -pix_fmt yuv420p "
        f"-f rtsp {output_stream}"
    ).split()
    return subprocess.Popen(args, stdin=subprocess.PIPE)

ffmpeg_process = open_ffmpeg_stream_process()

if grab_frame(cap) is None:
    force_exit = True
while not force_exit:
    time.sleep(1/fps)
    frame = grab_frame(cap)
    if frame is not None:
        inputs=[grpcclient.InferInput( args.input_name, [1,WIDTH,HEIGHT,3], "FP32")]
        inputs[0].set_data_from_numpy(np.array([frame], dtype=np.float32))
        triton_client.async_infer(
            model_name=args.model_name,
            callback=partial(callback, frame),
            inputs=inputs)

finish()
# When everything done, release the capture
cap.release()
ffmpeg_process.kill()
print(f"Finished.")
