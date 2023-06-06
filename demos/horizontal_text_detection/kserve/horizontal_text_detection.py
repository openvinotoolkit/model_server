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

parser = argparse.ArgumentParser()
parser.add_argument('--grpc_address', required=False, default='localhost', help='Specify url to grpc service')
parser.add_argument('--grpc_port', required=False, default=8080, type=int, help='Specify port to grpc service')
parser.add_argument('--video_source', required=False, default="0", type=str, help='Camera ID number or path to a video file')
args = parser.parse_args()

address = "{}:{}".format(args.grpc_address, args.grpc_port)
triton_client = grpcclient.InferenceServerClient( url=address, verbose=False)

try:
    source = int(args.video_source)
except ValueError:
    source = args.video_source

cap = cv2.VideoCapture('tcpclientsrc port=8080 host=localhost ! tsdemux ! h264parse ! avdec_h264 ! videoconvert ! appsink')
force_exit = False

out = cv2.VideoWriter("appsrc ! videoconvert ! video/x-raw,width=704,height=704 ! x264enc threads=1 ! mpegtsmux ! tcpserversink port=8088 host=0.0.0.0", 0, 30, (704, 704), True)

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
    #print(output)
    preds = output.argmax(2)
    word = ''
    for i in range(preds.shape[0]):
        word += alphabet[preds[i,0]]
    return decode(word)

def finish():
    global force_exit
    force_exit = True

def grab_frame(cap):
    WIDTH = 704
    HEIGHT = 704
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


def callback(frame, result, error):
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
    out.write(frame)
    if error is not None:
        print(error)


if grab_frame(cap) is None:
    print("[ERROR] Check camera input...")
    force_exit = True

while not force_exit:
    time.sleep(0.3)
    frame = grab_frame(cap)
    inputs=[grpcclient.InferInput( "image", [1,704,704,3], "FP32")]
    inputs[0].set_data_from_numpy(np.array([frame], dtype=np.float32))
    triton_client.async_infer(
        model_name="detect_text_images",
        callback=partial(callback, frame),
        inputs=inputs)

finish()
# When everything done, release the capture
cap.release()
print(f"Finished.")
