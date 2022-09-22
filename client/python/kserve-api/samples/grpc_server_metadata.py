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

import argparse
import tritonclient.grpc as grpcclient

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sends request via KServe gRPC API to get server metadata.')
    parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to gRPC service. default:localhost')
    parser.add_argument('--grpc_port',required=False, default=5000, help='Specify port to gRPC service. default: 9000')

    args = vars(parser.parse_args())

    address = "{}:{}".format(args['grpc_address'],args['grpc_port'])

    client = grpcclient.InferenceServerClient(address)
    print(client.get_server_metadata())

