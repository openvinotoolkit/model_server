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

import time
import cv2
from functools import partial
import queue
import threading
import tritonclient.grpc as grpcclient
import subprocess #nosec
import numpy as np

class Datatype:
    def dtype(self):
        pass
    def string(self):
        pass

class FP32(Datatype):
    def dtype(self):
        return np.float32
    def string(self):
        return "FP32"

class UINT8(Datatype):
    def dtype(self):
        return np.uint8
    def string(self):
        return "UINT8"

class OutputBackend:
    def init(self, sink, fps, width, height):
        pass
    def write(self, frame):
        pass
    def release(self):
        pass

class FfmpegOutputBackend(OutputBackend):
    def init(self, sink, fps, width, height, bitrate : int = 1000):
        args = (
            "ffmpeg -re -stream_loop -1 -f rawvideo -err_detect aggressive -fflags discardcorrupt -pix_fmt "
                f"bgr24 -r {fps} -s {width}x{height} -i pipe:0 -cpu-used 6 -avioflags direct -deadline realtime -pix_fmt yuv420p -c:v libvpx -muxdelay 0.1 "
            f"-b:v {bitrate}k -f rtsp {sink}"
        ).split()
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE) #nosec
    def write(self, frame):
        self.process.stdin.write(frame.astype(np.uint8).tobytes())
    def release(self):
        self.process.kill()

class CvOutputBackend(OutputBackend):
    def init(self, sink, fps, width, height):
        self.cv_sink = cv2.VideoWriter(sink, cv2.VideoWriter_fourcc(*'avc1'), fps, (width,height))
    def write(self, frame):
        self.cv_sink.write(frame)
    def release(self):
        self.cv_sink.release()

class ImshowOutputBackend(OutputBackend):
    def init(self, sink, fps, width, height):
        ...
    def write(self, frame):
        cv2.imshow("OVMS StreamClient", frame)
        cv2.waitKey(1)
    def release(self):
        cv2.destroyAllWindows()

class StreamClient:
    class OutputBackends():
        ffmpeg = FfmpegOutputBackend()
        cv2 = CvOutputBackend()
        imshow = ImshowOutputBackend()
        none = OutputBackend()
    class Datatypes():
        fp32 = FP32()
        uint8 = UINT8()

    def __init__(self, *, preprocess_callback = None, postprocess_callback, source, sink: str, ffmpeg_output_width = None, ffmpeg_output_height = None, output_backend: OutputBackend = OutputBackends.ffmpeg, verbose: bool = False, exact: bool = True, benchmark: bool = False, max_inflight_packets: int = 4):
        """
        Parameters
        ----------
        preprocess_callback
            Function used to prepare input image for inference.
        postprocess_callback
            Function used to merge inference results with original image. For mediapipe use only inference output.
        source : any
            RTSP address, filepath or camera id used as framesource for inference.
        sink : str
            RTSP address in case of ffmpeg backend, filepath in case of cv2 backend.
        output_backend : OutputBackend
            Backed used for presenting postprocessed frames.
        verbose : bool
            Should client output debug information.
        exact : bool
            Should client push every frame into output backwend.
        benchmark : bool
            Should client collect processing times
        """

        self.preprocess_callback = preprocess_callback
        self.postprocess_callback = postprocess_callback
        self.force_exit = False
        self.source = source
        self.sink = sink
        self.width = ffmpeg_output_width
        self.height = ffmpeg_output_height
        self.output_backend = output_backend
        self.verbose = verbose
        self.exact = exact
        self.benchmark = benchmark

        self.pq = queue.PriorityQueue()
        self.req_q = queue.Queue(max_inflight_packets)

    def grab_frame(self):
        success, frame = self.cap.read()
        if not success:
            if self.verbose:
                print("[WARNING] No Input frame")
            self.force_exit = True
            return None

        if self.preprocess_callback is not None:
            return self.preprocess_callback(frame)
        else:
            return frame

    inference_time = []
    dropped_frames = 0
    frames = 0
    def callback(self, frame, i, timestamp, result, error):
        if error is not None:
            if self.benchmark:
                self.dropped_frames += 1
            if self.verbose:
                print(error)
        if i == None:
            i = result.get_response().parameters["OVMS_MP_TIMESTAMP"].int64_param
        if timestamp == None:
            timestamp = result.get_response().parameters["OVMS_MP_TIMESTAMP"].int64_param
        frame = self.postprocess_callback(frame, result)
        self.pq.put((i, frame, timestamp))
        self.req_q.get()

    def display(self):
        i = 0 
        while True:
            entry = self.pq.get()
            if (entry[0] == i and self.exact and self.streaming_api is not True) or (entry[0] > i and (self.exact is not True or self.streaming_api is True)):
                if isinstance(entry[1], str) and entry[1] == "EOS":
                    break
                frame = entry[1]
                if frame is not None:
                    self.output_backend.write(frame)
                    if self.benchmark:
                        self.inference_time.insert(i, self.get_timestamp() - entry[2])
                        self.frames += 1
                if self.exact:
                    i += 1
                else:
                    if self.benchmark:
                        self.dropped_frames += entry[0] - i
                    i = entry[0]
            elif self.exact:
                self.pq.put(entry)

    def get_timestamp(self) -> int:
        return int(cv2.getTickCount() / cv2.getTickFrequency() * 1e6)

    def start(self, *, ovms_address : str, input_name : str, model_name : str, datatype : Datatype = FP32(), batch = True, limit_stream_duration : int = 0, limit_frames : int = 0, streaming_api: bool = False):
        """
        Parameters
        ----------
        ovms_address : str
            Address for inference
        input_name : str
            Name of the model's input
        model_name : str
            Namoe of the model
        datatype : Datatype
            Input type of loaded model
        batch : bool
            Determines if client should reserve shape dimension for batching
        limit_stream_duration : int
            Limits how long client could run
        limit_frames : int
            Limits how many frames should be processed
        streaming_api : bool
            Use experimental streaming endpoint
        """

        self.cap = cv2.VideoCapture(int(self.source) if len(self.source) == 1 and self.source[0].isdigit() else self.source, cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Starting grpc client {ovms_address}")
        triton_client = grpcclient.InferenceServerClient(url=ovms_address, verbose=False)
        self.streaming_api = streaming_api

        display_th = threading.Thread(target=self.display)
        display_th.start()
        test_frame = self.grab_frame()
        np_test_frame = np.array(test_frame, dtype=datatype.dtype())
        if test_frame is None:
            self. force_exit = True
        else:
            if self.width is None:
                self.width = np_test_frame.shape[1]
            if self.height is None:
                self.height = np_test_frame.shape[0]
        self.output_backend.init(self.sink, fps, self.width, self.height)

        if streaming_api:
            triton_client.start_stream(partial(self.callback, None, None, None))
            
        frame_number = 0
        total_time_start = time.time()
        try:
            while not self.force_exit:
                self.req_q.put(frame_number)
                timestamp = time.time()
                frame = self.grab_frame()
                if frame is not None:
                    np_frame = np.array([frame], dtype=datatype.dtype()) if batch else np.array(frame, dtype=datatype.dtype())
                    inputs=[grpcclient.InferInput(input_name, np_frame.shape, datatype.string())]
                    inputs[0].set_data_from_numpy(np_frame)
                    if streaming_api:
                        triton_client.async_stream_infer(model_name=model_name, inputs=inputs, parameters={"OVMS_MP_TIMESTAMP":self.get_timestamp()}, request_id=str(frame_number))
                    else:
                        triton_client.async_infer(
                            model_name=model_name,
                            callback=partial(self.callback, frame, frame_number, timestamp),
                            inputs=inputs)
                    frame_number += 1
                if limit_stream_duration > 0 and time.time() - total_time_start > limit_stream_duration:
                    break
                if limit_frames > 0 and frame_number > limit_frames:
                    break
        finally:
            self.pq.put((frame_number, "EOS"))
            if streaming_api:
                triton_client.stop_stream()
        sent_all_frames = time.time() - total_time_start


        self.cap.release()
        display_th.join()
        self.output_backend.release()
        total_time = time.time() - total_time_start
        if self.benchmark:
            print(f"{frame_number} {len(self.inference_time)}")
            print(f"{{\"average_inference_latency\": {sum(self.inference_time)/len(self.inference_time)/1e6}, \"dropped_frames\": {self.dropped_frames}, \"frames\": {self.frames}, \"fps\": {self.frames/total_time}, \"total_time\": {total_time}, \"sent_all_frames\": {sent_all_frames}}}")
