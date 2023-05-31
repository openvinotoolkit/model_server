#
# Copyright (c) 2022 Intel Corporation
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sends requests via KServe gRPC API using images in numpy format. '
                                                 'It displays performance statistics and optionally the model accuracy')
    parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
    parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')

    args = vars(parser.parse_args())

    address = "{}:{}".format(args['grpc_address'],args['grpc_port'])

    triton_client = grpcclient.InferenceServerClient(
                url=address,
                verbose=False)

    inputs = []
    inputs.append(grpcclient.InferInput("in1", [1,10], "FP32"))
    inputs.append(grpcclient.InferInput("in2", [1,10], "FP32"))
    inputs[0].set_data_from_numpy(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.float32))
    inputs[1].set_data_from_numpy(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.float32))
    outputs = []
    results = triton_client.infer(
        model_name= "dummyAdd",
        inputs=inputs,
        outputs=outputs)
    output = results.as_numpy("out")
    print('Output:')
    print(output)