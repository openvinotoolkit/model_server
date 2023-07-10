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
import queue
import threading
import tritonclient.grpc as grpcclient
import numpy as np
import subprocess #nosec

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
cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
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

raw_frame = np.empty((HEIGHT, WIDTH, 3), np.uint8) 
frame_bytes = memoryview(raw_frame).cast("B")
def grab_frame(cap):
    success, frame = cap.read()
    if not success:
        print("[WARNING] No Input frame")
        finish()
        return None

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

pq = queue.PriorityQueue()

def callback(frame, i, result, error):
    frame = postprocess(frame, result)
    # ffmpeg_process.stdin.write(frame.astype(np.uint8).tobytes())
    pq.put((i, frame))
    if error is not None:
        print(error)


def open_ffmpeg_stream_process():
    args = (
        "ffmpeg -re -stream_loop -1 -f rawvideo -pix_fmt "
        f"bgr24 -r {fps} -s {WIDTH}x{HEIGHT} -i pipe:0 -pix_fmt yuv420p "
        f"-tune zerolatency  -fflags nobuffer -b:v 1000k -preset ultrafast -f rtsp {output_stream}"
    ).split()
    return subprocess.Popen(args, stdin=subprocess.PIPE) #nosec

ffmpeg_process = open_ffmpeg_stream_process()

def display():
    i = 0 
    while True:
        if pq.qsize() > 0:
            print(pq.qsize())
        if pq.empty():
            continue
        if pq.queue[0][0] == i:
            frame = pq.get()[1]
            ffmpeg_process.stdin.write(frame.astype(np.uint8).tobytes())
            i += 1

display_th = threading.Thread(target=display)
display_th.start()
if grab_frame(cap) is None:
    force_exit = True
i = 0
while not force_exit:
    frame = grab_frame(cap)
    if frame is not None:
        inputs=[grpcclient.InferInput( args.input_name, [1,WIDTH,HEIGHT,3], "FP32")]
        inputs[0].set_data_from_numpy(np.array([frame], dtype=np.float32))
        triton_client.async_infer(
            model_name=args.model_name,
            callback=partial(callback, frame, i),
            inputs=inputs)
        i += 1

finish()
# When everything done, release the capture
cap.release()
ffmpeg_process.kill()
print(f"Finished.")
