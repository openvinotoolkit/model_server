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
from threading import Event
import datetime
from io import BytesIO
import argparse
import numpy as np
from PIL import Image
import cv2
import queue

parser = argparse.ArgumentParser(description='Client for stable diffusion example')

parser.add_argument('--url', required=False, default='localhost:9000',
                    help='Specify url to grpc service. default:localhost:9000')
parser.add_argument('--prompt',
                    required=False,
                    default='Zebras in space',
                    help='Prompt for image generation')

args = vars(parser.parse_args())

channel_args = [
    # Do not drop the connection for long workloads
    ("grpc.http2.max_pings_without_data", 0),
]
client = grpcclient.InferenceServerClient(args['url'], channel_args=channel_args)
data = args['prompt'].encode()

image_queue = queue.Queue()
event = Event()
videodims = (512,512)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
video = cv2.VideoWriter("image.mp4",fourcc, 4,videodims)

def callback(result, error):
    global image_queue, event
    if error:
        raise error
    if result.as_numpy('end_signal') is not None:
        image_queue.put(None)
    elif result.as_numpy('image') is not None:
        img = Image.open(BytesIO(result.as_numpy("image")))
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        image_queue.put(opencvImage)

client.start_stream(callback=callback)

model_name = "python_model"
input_name = "text"

start = datetime.time()
infer_input = grpcclient.InferInput(input_name, [len(data)], "BYTES")
infer_input._raw_content = data

start_time = datetime.datetime.now()
client.async_stream_infer(model_name, [infer_input])

print("Executing pipeline")
i = 0
while True:
    image = image_queue.get()
    if image is None:
        break
    cv2.imshow("image",image)
    cv2.waitKey(25)
    cv2.imwrite("image"+str(i)+".png", image)
    video.write(image)
    i += 1
endtime = datetime.datetime.now()
client.stop_stream()
video.release()
print("Generated final image", "image" + str(i) + ".png")
print("Transition saved to image.mp4")
print("Total time", int((endtime - start_time).total_seconds() * 1000), "ms")
print("Number of responses", i)
print('Average response time: {:.2f} ms'.format(int((endtime - start_time).total_seconds() * 1000)/i))
cv2.destroyAllWindows() 
