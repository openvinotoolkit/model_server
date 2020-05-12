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
import numpy as np

from src.logger import get_logger
from src.api.models.model import Model
from src.api.types import Tag, Attribute, SingleClassification, Classification
from src.preprocessing.preprocess_image import preprocess_binary_image as default_preprocessing

logger = get_logger(__name__)


class ClassificationAttributes(Model):  

    def postprocess_inference_output(self, inference_output: dict) -> str:
        # model with output shape for each classification output_name (1,N,1,1) 
        classifications = []
        for output_name in self.labels.keys():
            attributes = []
            highest_prob = 0.0

            if output_name not in inference_output:
                message = 'Model configuration label- {}'
                ' not found in model output - {}'.format(output_name, inference_output)
                logger.exception(message)
                raise ValidationError(message)

            for class_id in self.labels[output_name].keys():
                class_name = self.labels[output_name][class_id]
                probability = inference_output[output_name][0,int(float(class_id)),0,0].item()
                if probability > highest_prob:
                    tag_name = class_name 
                    highest_prob = probability
                attribute = Attribute(output_name, class_name, probability)
                attributes.append(attribute)

            classification = SingleClassification(attributes)
            classifications.append(classification)

        model_classification = Classification(subtype_name=self.model_name, classifications=classifications)

        response = json.dumps(model_classification.as_dict())

        return response
