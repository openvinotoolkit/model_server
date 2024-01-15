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
sys.path.append('../../common/stream_client/')
from stream_client import StreamClient

parser = argparse.ArgumentParser()
parser.add_argument('--grpc_address', required=False, default='localhost:9022', help='Specify url to grpc service')
parser.add_argument('--input_stream', required=False, default="rtsp://localhost:8080/channel1", type=str, help='Url of input rtsp stream')
parser.add_argument('--output_stream', required=False, default="rtsp://localhost:8080/channel2", type=str, help='Url of output rtsp stream')
parser.add_argument('--model_name', required=False, default="holisticTracking", type=str, help='Name of the model')
parser.add_argument('--input_name', required=False, default="first_input_video", type=str, help='Name of the model\'s input')
parser.add_argument('--verbose', required=False, default=False, help='Should client dump debug information', action='store_true')
parser.add_argument('--benchmark', required=False, default=False, help='Should client collect processing times', action='store_true')
parser.add_argument('--limit_stream_duration', required=False, default=0, type=int, help='Limit how long client should run')
parser.add_argument('--limit_frames', required=False, default=0, type=int, help='Limit how many frames should be processed')
args = parser.parse_args()

def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def postprocess(frame, result):
    frame = None
    if result is not None:
        frame = result.as_numpy("output")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

if args.output_stream[:4] == "rtsp":
    backend = StreamClient.OutputBackends.ffmpeg
    exact = False
elif args.output_stream == "screen":
    backend = StreamClient.OutputBackends.imshow
    exact = False
else:
    backend = StreamClient.OutputBackends.cv2
    exact = True

client = StreamClient(postprocess_callback = postprocess, preprocess_callback=preprocess, output_backend=backend, source=args.input_stream, sink=args.output_stream, exact=exact, benchmark=args.benchmark, verbose=args.verbose)
client.start(ovms_address=args.grpc_address, input_name=args.input_name, model_name=args.model_name, datatype = StreamClient.Datatypes.uint8, batch = False, limit_stream_duration = args.limit_stream_duration, limit_frames = args.limit_frames, streaming_api=True)

