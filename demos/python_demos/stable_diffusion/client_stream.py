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
import tritonclient.grpc as grpcclient
import threading
import time
from io import BytesIO
import argparse
import numpy as np
from PIL import Image, ImageDraw
import cv2


parser = argparse.ArgumentParser(description='Client for stable diffusion example')

parser.add_argument('--url', required=False, default='localhost:9000',
                    help='Specify url to grpc service. default:localhost:9000')
args = vars(parser.parse_args())

channel_args = [
    # Do not drop the connection for long workloads
    ("grpc.http2.max_pings_without_data", 0),
]
client = grpcclient.InferenceServerClient(args['url'], channel_args=channel_args)
data = "Zebras in space".encode()


videodims = (512,512)
#fourcc = cv2.VideoWriter_fourcc(*'avc1')    
fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
video = cv2.VideoWriter("test.mp4",fourcc, 4,videodims)
#img = Image.new('RGB', videodims, color = 'darkred')
#draw stuff that goes on every frame here
#for i in range(0,60*60):
#    imtemp = img.copy()
#    # draw frame specific stuff here.
#    video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
#video.release()

i = 0

def callback(result, error):
    global i
    i += 1
    if error:
        raise error
    elif result.as_numpy('OUTPUT') is not None:
        img = Image.open(BytesIO(result.as_numpy("OUTPUT")))
        img.save(f"output_{i}.png")
        imtemp = img.copy()
        video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
        print('got iteration', i)
    else:
        video.release()
        client.stop_stream()


client.start_stream(callback=callback)

model_name = "python_model"
input_name = "text"

start = time.time()
infer_input = grpcclient.InferInput(input_name, [len(data)], "BYTES")
infer_input._raw_content = data

client.async_stream_infer(model_name, [infer_input])
