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

parser = argparse.ArgumentParser(description='Client for seq2seq example')

parser.add_argument('--url', required=False, default='localhost:9000',
                    help='Specify url to grpc service. default:localhost:9000')
args = vars(parser.parse_args())

client = grpcclient.InferenceServerClient(args['url'])
text = "He never went out without a book under his arm, and he often came back with two."
print(f"Text:\n{text}\n")
data = text.encode()
infer_input = grpcclient.InferInput("text", [len(data)], "BYTES")
infer_input._raw_content = data
results = client.infer("python_model", [infer_input])
print(f"Translation:\n{results.as_numpy('OUTPUT').tobytes().decode()}\n")
