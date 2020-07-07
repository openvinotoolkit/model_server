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

import json
import sys
import cv2
from abc import ABC, abstractmethod

from commons import load_ov_standard_image, load_ams_image, \
    extract_content_type, send_image_ams, send_image_ovms, \
    run_ov_request


class Model(ABC):
    def __init__(self, ams_address, ams_port, ams_endpoint, ovms_address,
                 ovms_port, ovms_model_name, image_path, model_json):
        self.image_path = image_path
        self.ams_address = ams_address
        self.ams_port = ams_port
        self.ams_endpoint = ams_endpoint
        self.ovms_address = ovms_address
        self.ovms_port = ovms_port
        self.ovms_model_name = ovms_model_name
        self.parse_model_properties(model_json)
        self.ams_img_out = cv2.imread(image_path)
        self.ov_img_out = cv2.imread(image_path)
        self.ovms_img_out = cv2.imread(image_path)

    def parse_model_properties(self, model_json):
        try:
            with open(model_json, 'r') as jsonfile:
                model_cfg = json.loads(jsonfile.read())
        except Exception as e:
            print("Unable to load JSON configuration")
            sys.exit()
        # Only one input supported
        self.input_name = model_cfg['inputs'][0]['input_name']
        self.target_width = model_cfg['inputs'][0]['target_width']
        self.target_height = model_cfg['inputs'][0]['target_height']
        try:
            self.channels = model_cfg['inputs'][0]['channels']
        except KeyError:
            print(
                "Model config does not have channels configuration. Using default value 3")
            self.channels = 3
        self.output_names = []
        self.classes = {}
        self.not_softmax_outputs = []
        for out in model_cfg['outputs']:
            self.output_names.append(out['output_name'])
            if "classes" in out:
                self.classes[out['output_name']] = out["classes"]

    def input_preprocess(self):
        self.content_type = extract_content_type(self.image_path)
        if self.content_type is None:
            print("Invalid image file provided {}".format(self.image_path))
            sys.exit()
        self.ovms_image = load_ov_standard_image(self.image_path, self.target_height, self.target_width)
        self.ams_image = load_ams_image(self.image_path)

    def send_data(self):
        self.ov_output = run_ov_request(self.ovms_image, self.input_name, self.ovms_model_name)
        self.ovms_output = send_image_ovms(self.ovms_address, self.ovms_port,
                                           self.ovms_image, self.ovms_model_name, self.input_name, self.output_names,
                                           self.channels, self.target_height, self.target_width)
        self.ams_output = send_image_ams(self.ams_image, self.ams_address,
                                         self.ams_port, self.ams_endpoint, self.content_type)

    @abstractmethod
    def output_postprocess(self):
        pass

    @abstractmethod
    def print_results(self):
        pass
