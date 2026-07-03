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

import numpy as np
import datetime
import argparse
import tritonclient.grpc as grpcclient
from tritonclient.utils import serialize_byte_tensor


parser = argparse.ArgumentParser(description='Do requests to OpenVINO Model Server using strings in KServe gRPC format')
parser.add_argument('--grpc_address', required=False, default='localhost', help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port', required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--input_name', required=False, default='inputs', help='Specify input tensor name. default: inputs')
parser.add_argument('--output_name', required=False, default='outputs', help='Specify output name. default: outputs')
parser.add_argument('--model_name', default='usem', help='Define model name, must be same as is in service. default: usem')
parser.add_argument('--string', required=True, default='', help='String to query.')
args = vars(parser.parse_args())

client = grpcclient.InferenceServerClient(url="{}:{}".format(args['grpc_address'], args['grpc_port']))

data_bytes = serialize_byte_tensor(np.array([args['string']], dtype=np.object_)).item()
infer_input = grpcclient.InferInput(args['input_name'], [len(data_bytes)], "BYTES")
infer_input._raw_content = data_bytes

start_time = datetime.datetime.now()
result = client.infer(args['model_name'], [infer_input])
end_time = datetime.datetime.now()
duration = (end_time - start_time).total_seconds() * 1000
print("processing time", duration, "ms.")
output = result.as_numpy(args['output_name'])
print("Output shape", output.shape)
print("Output subset", output[0, :20])
