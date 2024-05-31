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
from tritonclient.utils import deserialize_bytes_tensor
import argparse
from threading import Event
import datetime
import numpy as np
import os
from utils import serialize_prompts

parser = argparse.ArgumentParser(description='Client for llm example')

parser.add_argument('--url', required=False, default='localhost:9000',
                    help='Specify url to grpc service. default:localhost:9000')
parser.add_argument('-p', '--prompt',
                    required=True,
                    default=[],
                    action="append",
                    help='Questions for the endpoint to answer')
args = vars(parser.parse_args())

channel_args = [
    # Do not drop the connection for long workloads
    ("grpc.http2.max_pings_without_data", 0),
]

client = grpcclient.InferenceServerClient(args['url'], channel_args=channel_args)

event = Event()
processing_times = np.zeros((0),int) # tracks response latency in ms

prompts = args['prompt']
completions = [f"==== Prompt: {prompts[i]} ====\n" for i in range(len(prompts))]
token_count = [0]
if len(prompts) == 1:
    print(f"Question:\n{prompts[0]}\n")

def callback(result, error):
    endtime = datetime.datetime.now()
    global event
    global start_time
    global processing_times
    if error:
        raise error
    if result.as_numpy('end_signal') is not None:
        event.set()
    elif result.as_numpy('token_count') is not None:
        token_count[0] = result.as_numpy('token_count')[0]
    elif result.as_numpy('completion') is not None:
        os.system('cls' if os.name=='nt' else 'clear')
        for i, completion in enumerate(deserialize_bytes_tensor(result._result.raw_output_contents[0])):
            completions[i] += completion.decode()
            print(completions[i])
            print()
        duration = int((endtime - start_time).total_seconds() * 1000)
        processing_times = np.append(processing_times, duration)
        start_time = datetime.datetime.now()
    else:
        assert False, "unexpected output"


client.start_stream(callback=callback)
infer_input = serialize_prompts(prompts)
start_time = datetime.datetime.now()
client.async_stream_infer(model_name="python_model", inputs=[infer_input])

event.wait()
client.stop_stream()
print('\nEND')


if token_count[0]>0:
    print("\n\nNumber of tokens ", token_count[0])
    print("Generated tokens per second ", round(token_count[0] / (np.sum(processing_times) / 1000), 2))
    print("Time per generated token", round((np.sum(processing_times) / 1000) / token_count[0] * 1000, 2), "ms")
print("Total time", np.sum(processing_times), "ms")
print("Number of responses", processing_times.size)
print("First response time", processing_times[0], "ms")
print('Average response time: {:.2f} ms'.format(np.average(processing_times)))
