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
from functools import partial
import queue
import threading
import tritonclient.grpc as grpcclient
import subprocess #nosec
from enum import Enum
import numpy as np

class StreamClient:
    class OutputBackend(Enum):
        ffmpeg = 1
        cv2 = 2
        other = 3
        none = 0
    def __init__(self, *, preprocess_callback = None, postprocess_callback, source, sink, ffmpeg_output_width = None, ffmpeg_output_height = None, output_backend = OutputBackend.ffmpeg, other_stream_sink = None, verbose = False):
        self.preprocess_callback = preprocess_callback
        self.postprocess_callback = postprocess_callback
        self.force_exit = False
        self.source = source
        self.sink = sink
        self.width = ffmpeg_output_width
        self.height = ffmpeg_output_height
        self.output_backend = output_backend
        self.stream_sink = other_stream_sink
        self.verbose = verbose

        self.pq = queue.PriorityQueue()

    def grab_frame(self):
        success, frame = self.cap.read()
        if not success:
            print("[WARNING] No Input frame")
            self.force_exit = True
            return None

        if self.preprocess_callback is not None:
            return self.preprocess_callback(frame)
        else:
            return frame

    def callback(self, frame, i, result, error):
        frame = self.postprocess_callback(frame, result)
        self.pq.put((i, frame))
        if error is not None and self.verbose == True:
            print(error)

    def open_ffmpeg_stream_process(self, fps):
        args = (
            "ffmpeg -re -stream_loop -1 -f rawvideo -err_detect aggressive -fflags discardcorrupt -pix_fmt "
                f"bgr24 -r {fps} -s {self.width}x{self.height} -i pipe:0 -cpu-used 6 -avioflags direct -deadline realtime -pix_fmt yuv420p -c:v libvpx -muxdelay 0.1 "
            f"-b:v 90k -f rtsp {self.sink}"
        ).split()
        return subprocess.Popen(args, stdin=subprocess.PIPE) #nosec

    def display(self):
        i = 0 
        while True:
            if self.pq.empty():
                continue
            entry = self.pq.get()
            if entry[0] > i:
                frame = entry[1]
                if self.output_backend.value == self.OutputBackend.ffmpeg.value:
                    self.ffmpeg_process.stdin.write(frame.astype(np.uint8).tobytes())
                elif self.output_backend.value == self.OutputBackend.cv2.value:
                    self.cv_sink.write(frame)
                elif self.output_backend.value == self.OutputBackend.other.value:
                    self.stream_sink(frame)

                i = entry[0]

    def start(self, *, ovms_address, input_name, model_name):
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        triton_client = grpcclient.InferenceServerClient(url=ovms_address, verbose=False)

        display_th = threading.Thread(target=self.display)
        display_th.start()
        test_frame = self.grab_frame()
        np_test_frame = np.array(test_frame, dtype=np.float32)
        if test_frame is None:
            self. force_exit = True
        else:
            if self.width is None:
                self.width = np_test_frame.shape[0]
            if self.height is None:
                self.height = np_test_frame.shape[1]
        if self.output_backend.value == self.OutputBackend.ffmpeg.value:
            self.ffmpeg_process = self.open_ffmpeg_stream_process(fps)
        elif self.output_backend.value == self.OutputBackend.cv2.value:
            self.cv_sink = cv2.VideoWriter(self.sink)
            
        i = 0
        while not self.force_exit:
            frame = self.grab_frame()
            if frame is not None:
                np_frame = np.array([frame], dtype=np.float32)
                inputs=[grpcclient.InferInput( input_name, np_frame.shape, "FP32")]
                inputs[0].set_data_from_numpy(np_frame)
                triton_client.async_infer(
                    model_name=model_name,
                    callback=partial(self.callback, frame, i),
                    inputs=inputs)
                i += 1

        self.cap.release()
        if self.output_backend.value == self.OutputBackend.ffmpeg.value:
            self.ffmpeg_process.kill()
        elif self.output_backend.value == self.OutputBackend.cv2.value:
            self.cv_sink.release()
        print(f"Finished.")
