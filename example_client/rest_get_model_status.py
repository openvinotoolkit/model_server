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
import json
import requests


parser = argparse.ArgumentParser(description='Get served model's status for RESTful API interface')
parser.add_argument('--rest_url', required=False, default='http://localhost',  help='Specify url to REST API service. default: http://localhost')
parser.add_argument('--rest_port', required=False, default=5555, help='Specify port to REST API service. default: 5555')
parser.add_argument('--model_name', default='resnet', help='Model name to query, must be the same as in the service. default: resnet',
                    dest='model_name')
parser.add_argument('--model_version', default=None, type=int, help='Model version to query - must be numerical. List all versions if omitted',
                    dest='model_version')
args = vars(parser.parse_args())

version = ""
if args.get('model_version') is not None:
    version = "/versions/{}".format(args.get('model_version'))
result = requests.get("{}:{}/v1/models/{}{}".format(args['rest_url'], args['rest_port'], args['model_name'], version))

try:
    result_dic = json.loads(result.text)
except ValueError:
    print("The server response is not json format: {}",format(result.text))
    exit(1)

print(result_dic)
