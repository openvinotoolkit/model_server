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
import argparse
import sys
sys.path.append('/')
from stream_client import StreamClient

parser = argparse.ArgumentParser()
parser.add_argument('--grpc_address', required=False, default='localhost:9000', help='Specify url to grpc service')
parser.add_argument('--input_stream', required=False, default="rtsp://localhost:8080/channel1", type=str, help='Url of input rtsp stream')
parser.add_argument('--output_stream', required=False, default="rtsp://localhost:8080/channel2", type=str, help='Url of output rtsp stream')
parser.add_argument('--model_name', required=False, default="detect_text_images", type=str, help='Name of the model')
parser.add_argument('--width', required=False, default=704, type=int, help='Width of model\'s input image')
parser.add_argument('--height', required=False, default=704, type=int, help='Height of model\'s input image')
parser.add_argument('--input_name', required=False, default="image", type=str, help='Name of the model\'s input')
parser.add_argument('--verbose', required=False, default=False, help='Should client dump debug information', action='store_true')
parser.add_argument('--benchmark', required=False, default=False, help='Should client collect processing times', action='store_true')
parser.add_argument('--limit_stream_duration', required=False, default=0, type=int, help='Limit how long client should run')
parser.add_argument('--limit_frames', required=False, default=0, type=int, help='Limit how many frames should be processed')
args = parser.parse_args()

WIDTH = args.width
HEIGHT = args.height



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
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz#'
    preds = output.argmax(2)
    word = ''
    for pred in preds:
        word += alphabet[pred[0]]
    return decode(word)

def preprocess(frame):
    if frame.shape[0] > HEIGHT and frame.shape[1] > WIDTH:
        frame = cv2.resize(frame, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)
    else:
        frame = cv2.resize(frame, (HEIGHT, WIDTH), interpolation=cv2.INTER_LINEAR)
    return frame

def postprocess(frame, result):
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

if args.output_stream[:4] == "rtsp":
    backend = StreamClient.OutputBackends.ffmpeg
    exact = False
else:
    backend = StreamClient.OutputBackends.cv2
    exact = True
client = StreamClient(postprocess_callback = postprocess, preprocess_callback=preprocess, output_backend=backend, source=args.input_stream, sink=args.output_stream, exact=exact, benchmark=args.benchmark)
client.start(ovms_address=args.grpc_address, input_name=args.input_name, model_name=args.model_name, limit_stream_duration = args.limit_stream_duration, limit_frames = args.limit_frames)

