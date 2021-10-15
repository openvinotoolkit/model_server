#
# Copyright (c) 2021 Intel Corporation
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
from ovmsclient import make_grpc_metadata_request, make_grpc_client

parser = argparse.ArgumentParser(description='Get information about the status of served models over gRPC interace')
parser.add_argument('--service_url', required=False, default='localhost:9000',
                    help='Specify url to grpc service. default:localhost', dest='service_url')
parser.add_argument('--model_name', default='resnet', help='Model name to query. default: resnet',
                    dest='model_name')
parser.add_argument('--model_version', default=0, type=int,
                    help='Model version to query. Lists all versions if omitted',
                    dest='model_version')
args = vars(parser.parse_args())

# configuration
service_url = args.get('service_url')   # default='localhost:9000'
model_name = args.get('model_name')  # default='resnet'
model_version = args.get('model_version')   # default=0

# creating grpc client
client = make_grpc_client(service_url)

# creating metadata request
request = make_grpc_metadata_request(model_name, model_version)

# getting model metadata from the server
metadata = client.get_model_metadata(request)
metadata_dict = metadata.to_dict()
print(metadata_dict)
