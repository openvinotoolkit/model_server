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

from src.api.models.vehicle_detection_adas_model import VehicleDetectionAdas
from src.api.ovms_connector import OvmsConnector

# Default version for a model is the latest one
DEFAULT_VERSION = 0 

AVAILABLE_MODELS = [
    {"name": "vehicleDetection", 
    "class": VehicleDetectionAdas, 
    "ovms_mapping": {
        "model_name": "vehicle_detection_adas",
        "model_version": DEFAULT_VERSION,
        "input_name": "data",
        "output_name": "detection_out",
        },
    "labels_path": "/opt/ams_models/vehicle_detection_adas_model.json"
    },
]
