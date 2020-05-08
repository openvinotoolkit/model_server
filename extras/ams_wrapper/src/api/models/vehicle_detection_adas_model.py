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
            # TODO: Error handling
            return
        return preprocessed_image                                                               


    def postprocess_inference_output(self, inference_output: dict) -> str:

        result_array = inference_output["detection_out"]

        # model with output shape (1,1,200,7) 
        # with last dimension containg detection details
        detections = []
        for detection in result_array[0][0]:
            label = str(detection[1].item())
            # End of detections
            if label == "0.0":
                break

            if not label in self.labels["detection_out"]:
                raise ValueError("label not found in labels definition")
            else:
                label_value = self.labels["detection_out"][label]

            image_id = detection[0].item()
            conf = detection[2].item()
            x_min = detection[3].item()
            y_min = detection[4].item()
            x_max = detection[5].item()
            y_max = detection[6].item()

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
