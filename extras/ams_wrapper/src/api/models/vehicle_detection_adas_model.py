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
import json
import numpy as np

from src.logger import get_logger
from src.api.models.model import Model
from src.api.types import Tag, Rectangle, SingleEntity, Entity
from src.preprocessing.preprocess_image import preprocess_binary_image as default_preprocessing

logger = get_logger(__name__)


class VehicleDetectionAdas(Model):   

    def preprocess_binary_image(self, binary_image: bytes) -> np.ndarray:
        try: 
            preprocessed_image = default_preprocessing(binary_image, target_size=(384,672))
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        except Exception as e:
            raise e
        return preprocessed_image                                                               


    def postprocess_inference_output(self, inference_output: dict) -> str:
        # Assuming single output
        output_config = next(iter(self.output_configs.values()))
        result_array = inference_output[output_config.output_name]

        # model with output shape (1,1,200,7) 
        # with last dimension containg detection details
        detections = []
        for detection in result_array[0][0]:
            label = detection[output_config.value_index_mapping['value']].item()
            # End of detections
            if label == 0.0:
                break

            if not label in self.labels[output_config.output_name]:
                raise ValueError("label not found in labels definition")
            else:
                label_value = self.labels[output_config.output_name][label]

            conf = detection[output_config.value_index_mapping['confidence']].item()
            x_min = detection[output_config.value_index_mapping['x_min']].item()
            y_min = detection[output_config.value_index_mapping['y_min']].item()
            x_max = detection[output_config.value_index_mapping['x_max']].item()
            y_max = detection[output_config.value_index_mapping['y_max']].item()

            tag = Tag(label_value, conf)

            box = Rectangle(x_min, y_min, abs(x_max-x_min), abs(y_max-y_min))

            detection = SingleEntity(tag, box)
            detections.append(detection)

        if len(detections) == 0:
            response = None
        else:
            entity = Entity(subtype_name=self.model_name, entities=detections)
            response = json.dumps(entity.as_dict())

        return response
