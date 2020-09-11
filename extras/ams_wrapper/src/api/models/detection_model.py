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
from src.api.types import Tag, Rectangle, SingleEntity, Entity, Detection


logger = get_logger(__name__)


class DetectionModel(Model):
    def postprocess_inference_output(self, inference_output: dict) -> str:
        # Assuming single output
        output_config: ModelOutputConfiguration = next(iter(self.output_configs.values()))
        output_name = output_config.output_name
        result_array = inference_output[output_name]

        # model with output shape (1,1,200,7)
        # with last dimension containg detection details
        detections = []
        entities = []
        for detection in result_array[0][0]:
            label = detection[output_config.value_index_mapping['value']].item()
            # End of detections
            if label == 0.0:
                break

            if label not in self.labels[output_name]:
                raise ValueError("label not found in labels definition")
            else:
                label_value = self.labels[output_name][label]

            conf = detection[output_config.value_index_mapping['confidence']].item()
            x_min = max(detection[output_config.value_index_mapping['x_min']].item(), 0)
            y_min = max(detection[output_config.value_index_mapping['y_min']].item(), 0)
            x_max = max(detection[output_config.value_index_mapping['x_max']].item(), 0)
            y_max = max(detection[output_config.value_index_mapping['y_max']].item(), 0)

            tag = Tag(label_value, conf)

            box = Rectangle(x_min, y_min, abs(x_max-x_min), abs(y_max-y_min))

            detection = SingleEntity(tag, box)
            detections.append(detection)

        detections.sort(key=lambda entity: entity.tag.confidence, reverse=True)
        if output_config.confidence_threshold:
            detections = [entity for entity in detections
                          if entity.tag.confidence >= output_config.confidence_threshold]
        if output_config.top_k_results:
            detections = detections[:output_config.top_k_results]

        if len(detections) == 0:
            response = None
        else:
            for detection in detections:
                entity = Entity(subtype_name=self.endpoint, entity=detection)
                entities.append(entity)
            model_detection = Detection(entities=entities)
            response = json.dumps(model_detection.as_dict())
        return response
