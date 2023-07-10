#
# Copyright (c) 2019 Intel Corporation
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
from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis import model_service_pb2_grpc
import grpc

def print_status_response(response):
    version_status = response.model_version_status
    for i in version_status:
        print("\nModel version: {}".format(i.version))
        print("State",state_names[i.state])
        print("Error code: ",i.status.error_code)
        print("Error message: ",i.status.error_message)

    return

state_names = {
    0: "UNKNOWN",
    10: "START",
    20: "LOADING",
    30: "AVAILABLE",
    40: "UNLOADING",
    50: "END"
}
#  https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/get_model_status.proto

parser = argparse.ArgumentParser(description='Get information about the status of served models over gRPC interface')
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--model_name', default='resnet', help='Model name to query. default: resnet',
                    dest='model_name')
parser.add_argument('--model_version', type=int, help='Model version to query. Lists all versions if omitted',
                    dest='model_version')
args = vars(parser.parse_args())

channel = grpc.insecure_channel("{}:{}".format(args['grpc_address'],args['grpc_port']))

stub = model_service_pb2_grpc.ModelServiceStub(channel)

print('Getting model status for model:',args.get('model_name'))

request = get_model_status_pb2.GetModelStatusRequest()
request.model_spec.name = args.get('model_name')
if args.get('model_version') is not None:
    request.model_spec.version.value = args.get('model_version')


result = stub.GetModelStatus(request, 10.0) # result includes a dictionary with all model outputs

print_status_response(response=result)
