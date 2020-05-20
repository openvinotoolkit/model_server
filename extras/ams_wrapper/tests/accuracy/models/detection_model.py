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

from models.model import Model
from commons import detection_array, detection_json


class DetectionModel(Model):

    def output_postprocess(self):
        self.ams_results = detection_json(self.ams_output, self.img_out,
                                          self.target_height, self.target_width)
        self.ovms_results = detection_array(self.ovms_output, self.img_out,
                                            self.target_height, self.target_width)

    def print_results(self):
        print(self.ams_results)
        print(self.ovms_results)
        if len(self.ams_results) == len(self.ovms_results):
            print("The same count of object has been found")
        else:
            print("Failed to find the same objects")
            return
