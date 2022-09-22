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
import tritonclient.http as httpclient

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sends request via KServe HTTP API to check if server is ready.')
    parser.add_argument('--http_address',required=False, default='localhost',  help='Specify url to HTTP service. default:localhost')
    parser.add_argument('--http_port',required=False, default=8000, help='Specify port to HTTP service. default: 8000')

    args = vars(parser.parse_args())

    address = "{}:{}".format(args['http_address'],args['http_port'])

    client = httpclient.InferenceServerClient(address)
    print("Server Ready: {}".format(client.is_server_ready()))

