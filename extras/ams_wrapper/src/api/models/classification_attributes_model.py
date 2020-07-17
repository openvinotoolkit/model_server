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

from src.logger import get_logger
from src.api.models.model import Model
from src.api.models.model_config import ModelOutputConfiguration
from src.api.types import Attribute, SingleClassification, Classification

logger = get_logger(__name__)


class ClassificationAttributes(Model):

    def postprocess_inference_output(self, inference_output: dict) -> str:
        # model with output shape for each classification output_name (1,N,1,1)
        classifications = []

        for output_name in self.labels.keys():
            attributes = []
            highest_prob = 0.0

            if output_name not in inference_output:
                message = 'Output name from model config - {} does not match model outputs - {}'.format(
                    output_name, inference_output)
                logger.exception(message)
                raise ValueError(message)

            # get output configuration for current output_name
            current_conf: ModelOutputConfiguration = self.output_configs[output_name]

            is_softmax = True
            value_multiplier = 1.0

            if not current_conf.is_softmax and current_conf.is_softmax is not None:
                is_softmax = current_conf.is_softmax
                value_multiplier = current_conf.value_multiplier

            for class_id in self.labels[output_name].keys():
                class_name = self.labels[output_name][class_id]
                probability = inference_output[output_name][0, int(
                    float(class_id)), 0, 0].item()
                if probability > highest_prob:
                    highest_prob = probability

                if is_softmax or is_softmax is None:
                    attribute = Attribute(class_name, probability)
                else:
                    value = probability * value_multiplier
                    attribute = Attribute(class_name, value)

                attributes.append(attribute)

            attributes.sort(key=lambda attr: attr.confidence, reverse=True)
            if current_conf.top_k_results:
                attributes = attributes[:current_conf.top_k_results]
            if current_conf.confidence_threshold:
                attributes = [attr for attr in attributes
                              if attr.confidence >= current_conf.confidence_threshold]
            classification = SingleClassification(subtype_name=output_name, attributes=attributes)
            classifications.append(classification)

        model_classification = Classification(classifications=classifications)

        response = json.dumps(model_classification.as_dict())

        return response
