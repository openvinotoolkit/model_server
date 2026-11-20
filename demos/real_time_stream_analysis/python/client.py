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

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

demo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(demo_root / 'common' / 'python'))
sys.path.insert(0, str(demo_root / 'common' / 'stream_client'))
from stream_client import StreamClient
import classes

parser = argparse.ArgumentParser()
parser.add_argument('--grpc_address', required=False, default='localhost:9000', help='Specify url to grpc service')
parser.add_argument('--ffmpeg_output_width', required=False, default=None, type=int, help='Width of the output video')
parser.add_argument('--ffmpeg_output_height', required=False, default=None, type=int, help='Height of the output video')
parser.add_argument('--input_stream', required=False, default="rtsp://localhost:8554/channel1", type=str, help='Url of input rtsp stream')
parser.add_argument('--output_stream', required=False, default="rtsp://localhost:8554/channel2", type=str, help='Url of output rtsp stream')
parser.add_argument('--model_name', required=False, default="resnetMediapipe", type=str, help='Name of the MediaPipe graph')
parser.add_argument('--input_name', required=False, default="in", type=str, help='Name of the graph input')
parser.add_argument('--output_name', required=False, default="out", type=str, help='Name of the graph output')
parser.add_argument('--verbose', required=False, default=False, help='Should client dump debug information', action='store_true')
parser.add_argument('--benchmark', required=False, default=False, help='Should client collect processing times', action='store_true')
parser.add_argument('--limit_stream_duration', required=False, default=0, type=int, help='Limit how long client should run')
parser.add_argument('--limit_frames', required=False, default=0, type=int, help='Limit how many frames should be processed')
args = parser.parse_args()

def preprocess(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.transpose(frame.astype(np.float32), (2, 0, 1))

def postprocess(frame, result):
    if frame is None or result is None:
        return frame

    scores = result.as_numpy(args.output_name)
    if scores is None or scores.size == 0:
        return frame

    class_index = int(np.argmax(scores))
    if scores.size == len(classes.imagenet_classes) + 1:
        label = "background" if class_index == 0 else classes.imagenet_classes.get(class_index - 1, str(class_index - 1))
    else:
        label = classes.imagenet_classes.get(class_index, str(class_index))
    label = label.split(",", 1)[0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(label, font, 0.8, 2)
    cv2.rectangle(frame, (8, 8), (20 + text_width, 20 + text_height + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, (14, 20 + text_height), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
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

client = StreamClient(postprocess_callback = postprocess, preprocess_callback=preprocess, output_backend=backend, source=args.input_stream, sink=args.output_stream, exact=exact, benchmark=args.benchmark, verbose=args.verbose, ffmpeg_output_width=args.ffmpeg_output_width, ffmpeg_output_height=args.ffmpeg_output_height)
client.start(ovms_address=args.grpc_address, input_name=args.input_name, model_name=args.model_name, datatype=StreamClient.Datatypes.fp32, batch=True, limit_stream_duration=args.limit_stream_duration, limit_frames=args.limit_frames, streaming_api=True)

