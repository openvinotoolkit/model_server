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
        self.ams_results = detection_json(self.ams_output, self.ams_img_out,
                                          self.target_height, self.target_width, self.ovms_model_name)
        self.ovms_results = detection_array(self.ovms_output, self.ovms_img_out,
                                            self.target_height, self.target_width, self.ovms_model_name, "ovms")
        self.ov_results = detection_array(self.ov_output, self.ov_img_out, self.target_height, self.target_width, self.ovms_model_name, "ov")

    def print_results(self):
        print("AMS detections: \n")
        for result in self.ams_results:
            print("confidence: {}".format(result[0]))
            print("box coordinates: {} {} {} {}\n".format(result[1], result[2], result[3], result[4]))
        print("OVMS detections: \n")
        for _, result in self.ovms_results.items():
            for res in result:
                print("confidence: {}".format(res[0]))
                print("box coordinates: {} {} {} {}\n".format(res[1], res[2], res[3], res[4]))
        print("OpenVINO detections: \n")
        for _, result in self.ov_results.items():
            for res in result:
                print("confidence: {}".format(res[0]))
                print("box coordinates: {} {} {} {}\n".format(res[1], res[2], res[3], res[4]))
