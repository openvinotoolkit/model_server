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
import os
import sys
import json

from src.logger import get_logger
from src.api.models.model import Model
from src.api.types import Tag, Attribute, SingleClassification, Classification

logger = get_logger(__name__)


class VehicleAttributes(Model):

    def postprocess_inference_output(self, inference_output: dict) -> str:

        # model with output shape for color (1,7,1,1) 
        # with second dimension containing colors
        # [white, gray, yellow, red, green, blue, black]
        # model with output shape for type (1,4,1,1) 
        # with second dimension containing types
        # [car, bus, truck, van]
        outputs = self.labels 
        classifications = []
        for output in outputs.keys():
            type_name = output
            attributes = []
            highest_prob = 0.0
            for position in outputs[output].keys():
                class_name = outputs[output][position]
                probability = inference_output[type_name][0,int(float(position)),0,0].item()
                if probability > highest_prob:
                    tag_name = class_name 
                    highest_prob = probability
                attribute = Attribute(type_name, class_name, probability)
                attributes.append(attribute)

            classification = SingleClassification(attributes)
            classifications.append(classification)

        model_classification = Classification(subtype_name=self.model_name, classifications=classifications)

        response = json.dumps(model_classification.as_dict())

        return response
