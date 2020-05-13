#
# Copyright (c) 2020 Intel Corporation
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

# should be run from model server directory
import argparse
import os
import urllib.request
from data.performance_constants import DATASET, IMAGE, ITERATIONS, AMS_PORT, AMS_ADDRESS

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', required=True, help='Name of the model to be used')
args = parser.parse_args()

for num in range(ITERATIONS):
    with open(os.path.join(DATASET, IMAGE), mode='rb') as image_file:
        image_bytes = image_file.read()
        url = 'http://{}:{}/{}'.format(AMS_ADDRESS, AMS_PORT, args.model_name)
        headers = {'Content-Type': 'image/png'}
        req = urllib.request.Request(url, image_bytes, headers=headers)
        response = urllib.request.urlopen(req)
        assert response.getcode() == 200, "Not expected response code: {}".format(response.getcode)
