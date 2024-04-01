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
import datetime
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
infer_input = serialize_prompts(args['prompt'])
start_time = datetime.datetime.now()
results = client.infer("python_model", [infer_input], client_timeout=10*60)  # 10 minutes
endtime = datetime.datetime.now()
if len(args['prompt']) == 1:
    print(f"Question:\n{args['prompt'][0]}\n\nCompletion:\n{results.as_numpy('completion').tobytes().decode()}\n")
else:
    for i, arr in enumerate(deserialize_bytes_tensor(results._result.raw_output_contents[0])):
        print(f"==== Prompt: {args['prompt'][i]} ====")
        print(arr.decode())
        print()

print("Total time", int((endtime - start_time).total_seconds() * 1000), "ms")
