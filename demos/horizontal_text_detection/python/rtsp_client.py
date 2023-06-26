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

import cv2
import time
import argparse
from functools import partial
import tritonclient.grpc as grpcclient
import numpy as np
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--grpc_address', required=False, default='localhost', help='Specify url to grpc service')
parser.add_argument('--grpc_port', required=False, default=9000, type=int, help='Specify port to grpc service')
parser.add_argument('--input_stream', required=False, default="rtsp://localhost:8080/channel1", type=str, help='Url of input rtsp stream')
parser.add_argument('--output_stream', required=False, default="rtsp://localhost:8080/channel2", type=str, help='Url of output rtsp stream')
parser.add_argument('--fps', required=False, default=30, type=int, help='Framerate of input stream')
args = parser.parse_args()
output_stream = args.output_stream

address = "{}:{}".format(args.grpc_address, args.grpc_port)
triton_client = grpcclient.InferenceServerClient( url=address, verbose=False)

WIDTH = 704
HEIGHT = 704
cap = cv2.VideoCapture(args.input_stream, cv2.CAP_FFMPEG)
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
    for i in range(preds.shape[0]):
        word += alphabet[preds[i,0]]
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

    # crop and resize if original image is too small or too big for the model
    if frame.shape[0] > HEIGHT and frame.shape[1] > WIDTH:
        frame = frame[0:HEIGHT, 0:WIDTH]
    else:
        res = min(frame.shape[0], frame.shape[1])
        frame = frame[0:res, 0:res]
        frame = cv2.resize(frame, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)
    return frame

def postprocess(frame, result):
    if result is not None:
        texts = result.as_numpy("texts")
        text_coordinates = result.as_numpy('text_coordinates')
        for i in range(len(texts)):
            text = get_text(texts[i])
            x_min = text_coordinates[i][0][0]
            y_min = text_coordinates[i][0][1]
            x_max = text_coordinates[i][0][0] + text_coordinates[i][0][2]
            y_max = text_coordinates[i][0][1] + text_coordinates[i][0][3]
            frame = cv2.rectangle(frame, (x_min,y_min), (x_max,y_max), (0,0,255), 1)
            cv2.putText(frame, text, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
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
    time.sleep(1/args.fps)
    frame = grab_frame(cap)
    inputs=[grpcclient.InferInput( "image", [1,WIDTH,HEIGHT,3], "FP32")]
    inputs[0].set_data_from_numpy(np.array([frame], dtype=np.float32))
    triton_client.async_infer(
        model_name="detect_text_images",
        callback=partial(callback, frame),
        inputs=inputs)

finish()
# When everything done, release the capture
cap.release()
print(f"Finished.")
