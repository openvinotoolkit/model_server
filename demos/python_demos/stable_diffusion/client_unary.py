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
import time
from PIL import Image
from io import BytesIO
import argparse

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

model_name = "python_model"
input_name = "text"

start = time.time()
infer_input = grpcclient.InferInput(input_name, [len(data)], "BYTES")
infer_input._raw_content = data

results = client.infer(model_name, [infer_input], client_timeout=10*60)
img = Image.open(BytesIO(results.as_numpy("image")))
img.save(f"output.png")
duration = time.time() - start
print("Generated image output.png")
print("Total response time: {} s".format(round(duration,2)))
