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

import numpy as np
import cv2
import grpc
import io
import threading
import time
import sys
import argparse

from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


parser = argparse.ArgumentParser()
parser.add_argument('--grpc_address', required=False, default='localhost', help='Specify url to grpc service')
parser.add_argument('--grpc_port', required=False, default=8080, type=int, help='Specify port to grpc service')
parser.add_argument('--num_threads', required=False, default=4, type=int, help='Number of threads for parallel service requesting')
parser.add_argument('--video_source', required=False, default="0", type=str, help='Camera ID number or path to a video file')
parser.add_argument('--use_case', required=False, default="text_spotting", type=str, choices=['text_spotting', 'ocr'], help='Which servable to request and how to postprocess')
args = parser.parse_args()

address = "{}:{}".format(args.grpc_address, args.grpc_port)
channel = grpc.insecure_channel(address)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

try:
    source = int(args.video_source)
except ValueError:
    source = args.video_source

cap = cv2.VideoCapture(source)
force_exit = False


def decode(text):
    word = ''
    last_character = None
    for character in text:
        if character == last_character:
            continue
        elif character == '_':
            last_character = None
        else:
            last_character = character
            word += character
    return word


def get_text(output):
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789_'
    preds = output.argmax(2)
    word = ''
    for i in range(preds.shape[0]):
        word += alphabet[preds[i,0]]
    return decode(word)


def draw_boxes_spotting(frame, result):
    output = make_ndarray(result.outputs['boxes'])
    for i in range(0, 100):  # there is returned 200 detections for each image in the batch
        detection = output[i,:]
        if detection[4] > 0.3:
            x_min = int(detection[0])
            y_min = int(detection[1])
            x_max = int(detection[2])
            y_max = int(detection[3])
            frame = cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,0,255),1)
    return frame


def draw_boxes_ocr(frame, result):
    texts = make_ndarray(result.outputs['texts'])
    text_coordinates = make_ndarray(result.outputs['text_coordinates'])
    for i in range(len(texts)):
        text = get_text(texts[i])
        x_min = text_coordinates[i][0][0]
        y_min = text_coordinates[i][0][1]
        x_max = text_coordinates[i][0][0] + text_coordinates[i][0][2]
        y_max = text_coordinates[i][0][1] + text_coordinates[i][0][3]
        frame = cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(36,255,12),1)
        cv2.putText(frame, text, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 1)
    return frame


class RequestingThread(threading.Thread):
    def __init__(self, index):
        print(f"Initializing requesting thread index: {index}")
        super().__init__()
        self.index = index
        self.input_frame = None
        self.output_frame = None
        self.predict_durations = []
        self.input_ready_event = threading.Event()
        self.output_ready_event = threading.Event()

    def is_initialized(self):
        return not (self.input_frame is None and self.output_frame is None)

    def wait_for_input(self):
        self.input_ready_event.wait()
        self.input_ready_event.clear()

    def wait_for_result(self):
        self.output_ready_event.wait()
        self.output_ready_event.clear()

    def notify_input_ready(self):
        self.input_ready_event.set()

    def notify_output_ready(self):
        self.output_ready_event.set()

    def set_input(self, frame):
        self.input_frame = frame
        self.notify_input_ready()

    def get_output(self):
        return self.output_frame

    def get_average_latency(self):
        return np.average(np.array(self.predict_durations))

    def run(self):
        print(f"Launching requesting thread index: {self.index}")
        global force_exit
        while (True):
            self.wait_for_input()
            if force_exit:
                break
            is_success, buffer = cv2.imencode(".jpg", self.input_frame)
            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'detect_text_images' if args.use_case == 'ocr' else 'text'
            request.inputs['image'].CopyFrom(make_tensor_proto([buffer.tobytes()], shape=[1]))

            predict_start_time = time.time()
            result = None
            try:
                result = stub.Predict(request, 10.0)
            except grpc.RpcError as err:
                if err.code() == grpc.StatusCode.ABORTED:
                    pass
                else:
                    raise err

            predict_duration = time.time() - predict_start_time
            predict_duration *= 1000
            self.predict_durations.append(predict_duration)

            if result is None:
                self.output_frame = np.array(self.input_frame, copy=True)
            else:
                if args.use_case == 'ocr':
                    self.output_frame = draw_boxes_ocr(self.input_frame, result)
                else:
                    self.output_frame = draw_boxes_spotting(self.input_frame, result)
            self.notify_output_ready()
        print(f"Stopping requesting thread index: {self.index}")


threads = [RequestingThread(i) for i in range(args.num_threads)]

for thread in threads:
    thread.start()


def grab_frame(cap):
    WIDTH = 704
    HEIGHT = 704
    ret, frame = cap.read()
    # crop and resize if original image is too small or too big for the model
    if frame.shape[0] > HEIGHT and frame.shape[1] > WIDTH:
        frame = frame[0:HEIGHT, 0:WIDTH]
    else:
        res = min(frame.shape[0], frame.shape[1])
        frame = frame[0:res, 0:res]
        frame = cv2.resize(frame, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)
    return frame


i = 0
frames_processed = 0
last_display_time = time.time()
app_start_time = time.time()

while(True):
    if not threads[i].is_initialized():
        threads[i].set_input(grab_frame(cap))
        i = (i + 1) % args.num_threads
        continue

    threads[i].wait_for_result()
    avg_latency_for_thread = threads[i].get_average_latency()
    frame_to_display = threads[i].get_output()
    threads[i].set_input(grab_frame(cap))

    cv2.imshow('frame', frame_to_display)
    now = time.time()
    time_since_last_display = now - last_display_time
    last_display_time = now

    frames_processed += 1

    current_fps = 1 / (time_since_last_display if time_since_last_display > 0 else 1)
    avg_fps = 1 / ((now - app_start_time) / frames_processed)
    
    print(f"ThreadID: {i:3}; Current FPS: {current_fps:8.2f}; Average FPS: {avg_fps:8.2f}; Average latency: {avg_latency_for_thread:8.2f}ms")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        force_exit = True
        for thread in threads:
            thread.notify_input_ready()
            thread.join()
        break

    i = (i + 1) % args.num_threads


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
