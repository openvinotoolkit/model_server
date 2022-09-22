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

import grpc
import argparse

from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sends requests via KServe gRPC API to check if model is ready for inference.')
    parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
    parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
    parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. default: resnet',
                        dest='model_name')
    parser.add_argument('--model_version', default="",
                        help='Define model version. If not specified, the default version will be taken from model server',
                        dest='model_version')

    args = vars(parser.parse_args())

    address = "{}:{}".format(args['grpc_address'],args['grpc_port'])

    channel = grpc.insecure_channel(address)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
    request = service_pb2.ModelReadyRequest(name=args.get("model_name"),
                                               version=args.get("model_version"))
    response = grpc_stub.ModelReady(request)
    print("Model Ready: {}".format(response.ready))

