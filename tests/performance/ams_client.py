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

import os
import urllib.request

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATASET = os.path.join(ROOT_PATH, "tests", "functional", "fixtures", "test_images", "performance")
IMAGE = "single_car_small_reshaped.png"

ITERATIONS = 1000

for num in range(ITERATIONS):
    with open(os.path.join(DATASET, IMAGE), mode='rb') as image_file:
        image_bytes = image_file.read()
        url = 'http://localhost:5000/vehicleDetection'
        headers = {'Content-Type': 'image/png'}
        req = urllib.request.Request(url, image_bytes, headers=headers)
        response = urllib.request.urlopen(req)
        assert response.getcode() == 200, "Not expected response code: {}".format(response.getcode)
