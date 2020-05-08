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

import numpy as np

from src.preprocessing import preprocess_binary_image
from src.api.models.model import Model

class ExampleModel(Model):

    # TODO: think how to handle multiple different inputs
    def preprocess_binary_image(self, binary_image: bytes, input_name: str) -> np.ndarray:
        preprocessing_config = self.input_configs[input_name]
        return preprocess_binary_image(image=binary_image, **preprocessing_config)

    def postprocess_inference_output(self, inference_output: dict) -> str:
       """
        Examplary flow:

        from api.types import Tag, Rectangle, SingleEntity, Entity

        result_array = inference_output[output_name]

        # assuming detection model with output shape (1,1,N,7) 
        # with last dimension containg detection details
        # TODO: add handling of empty rows

        detections = []
        for detection in result_array[0][0]:
            image_id = detection[0]
            label = detection[1]
            conf = detection[2]
            x_min = detection[3]
            y_min = detection[4]
            x_max = detection[5]
            y_max = detection[6]

            tag = Tag(self.labels[label], conf)

            box = Rectangle(x_min, y_min, abs(x_max-x_min), abs(y_max-y_min))

            detection = SingleEntity(tag, box)
            detections.append(detection)
        
        entity = Entity(subtype_name=self.model_name, entities=detections)
        response_dict = []
        response_dict[inferences] = [entity.as_dict()]
        response = json.dumps(response_dict)
        return response
       """
       return
