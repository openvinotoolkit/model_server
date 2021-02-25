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


parser = argparse.ArgumentParser(description='Get information about served models')
parser.add_argument('--rest_url', required=False, default='http://localhost',  help='Specify url to REST API service. default: http://localhost')
parser.add_argument('--rest_port', required=False, default=5555, help='Specify port to REST API service. default: 5555')
parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. default: resnet',
                    dest='model_name')
parser.add_argument('--model_version', default=None, type=int, help='Define model version - must be numerical',
                    dest='model_version')
parser.add_argument('--client_cert', required=False, default=None, help='Specify mTLS client certificate file. Default: None.')
parser.add_argument('--client_key', required=False, default=None, help='Specify mTLS client key file. Default: None.')
parser.add_argument('--ignore_server_verification', required=False, action='store_true', help='Skip TLS host verification. Do not use in production. Default: False.')
parser.add_argument('--server_cert', required=False, default=None, help='Path to a custom directory containing trusted CA certificates, server certificate, or a CA_BUNDLE file. Default: None, will use default system CA cert store.')

args = vars(parser.parse_args())

certs = None
verify_server = None
if args.get('client_cert') is not None or args.get('client_key') is not None:
  if args.get('client_cert') is not None and args.get('client_key') is not None and args.get('rest_url').startswith("https"):
    certs = (args.get('client_cert'), args.get('client_key'))
    if args.get('server_cert') is not None:
      verify_server = args.get('server_cert')
    if args.get('ignore_server_verification') is True:
      verify_server = False
  else:
    print("Error: in order to use mTLS, you need to provide both --client_cert and --client_key. In addition, your --rest_url flag has to begin with 'https://'.")
    exit(1)

version = ""
if args.get('model_version') is not None:
    version = "/versions/{}".format(args.get('model_version'))
result = requests.get("{}:{}/v1/models/{}{}/metadata".format(args['rest_url'], args['rest_port'], args['model_name'], version), cert=certs, verify=verify_server)

try:
    result_dic = json.loads(result.text)
except ValueError:
    print("The server response is not json format: {}",format(result.text))
    exit(1)

print(result.text)

