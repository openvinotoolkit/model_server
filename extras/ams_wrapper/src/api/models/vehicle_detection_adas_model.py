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
from logger import get_logger
from api.models.model import Model
from api.types import Tag, Rectangle, SingleEntity, Entity

logger = get_logger(__name__)



class VehicleDetectionAdas(Model):

    def load_default_labels(self):
        labels_path = os.path.abspath(__file__).replace(".py",".json")
        try:                                                                          
            with open(labels_path, 'r') as labels_file:                               
                data = json.load(labels_file)                                         
                self.labels = data
        except Exception as e:                                                        
            logger.exception("Error occurred while opening labels file: {}".format(e))
            sys.exit(1)                                                               
        return                                                                   


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

            if not label in self.labels:
                raise ValueError("label not found in labels definition")
            else:
                label_value = self.labels[label]

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
        
        entity = Entity(subtype_name=self.model_name, entities=detections)

        if len(detections) == 0:
            response = None
        else:
            response = json.dumps(entity.as_dict())

        return response
