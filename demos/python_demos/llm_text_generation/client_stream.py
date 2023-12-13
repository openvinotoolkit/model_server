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
import argparse
from threading import Event

parser = argparse.ArgumentParser(description='Client for llm example')

parser.add_argument('--url', required=False, default='localhost:9000',
                    help='Specify url to grpc service. default:localhost:9000')
parser.add_argument('--prompt',
                    required=False,
                    default='Describe the state of the healthcare industry in the United States in max 2 sentences',
                    help='Question for the endpoint to answer')
args = vars(parser.parse_args())

channel_args = [
    # Do not drop the connection for long workloads
    ("grpc.http2.max_pings_without_data", 0),
]

client = grpcclient.InferenceServerClient(args['url'], channel_args=channel_args)

event = Event()

def callback(result, error):
    global event
    if error:
        raise error
    if result.as_numpy('END_SIGNAL') is not None:
        event.set()
    elif result.as_numpy('OUTPUT') is not None:
        print(result.as_numpy('OUTPUT').tobytes().decode(), flush=True, end='')
    else:
        assert False, "unexpected output"


client.start_stream(callback=callback)

text = args['prompt']
print(f"Question:\n{text}\n")
data = text.encode()
infer_input = grpcclient.InferInput("pre_prompt", [len(data)], "BYTES")
infer_input._raw_content = data

client.async_stream_infer(model_name="python_model", inputs=[infer_input])

event.wait()
client.stop_stream()
print('\nEND')
