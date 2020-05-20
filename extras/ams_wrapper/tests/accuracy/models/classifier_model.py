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
from commons import classification_array, classification_json


class ClassifierModel(Model):

    def output_postprocess(self):
        self.ams_results = classification_json(self.ams_output)
        self.ovms_results = classification_array(
            self.ovms_output, self.output_names, self.classes)

    def print_results(self):
        print(self.ams_results)
        print(self.ovms_results)
        for result in self.ams_results:
            if self.ovms_results[result]["value"] == self.ams_results[result]["value"]:
                print("Class ID is correct")
            if abs(self.ams_results[result]["confidence"] - self.ovms_results[result]["confidence"]) < 0.02:
                print("Confidence is correct")
