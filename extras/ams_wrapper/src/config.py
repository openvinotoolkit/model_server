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

from api.models.example_model import ExampleModel
from api.ovms_connector import OvmsConnector
OVMS_PORT = 9000

# Default version for a model is the latest one
DEFAULT_VERSION = -1 

AVAILABLE_MODELS = [
    {"name": "model", 
    "class": ExampleModel, 
    "ovms_mapping": {
        "model_name": "model",
        "model_version": DEFAULT_VERSION,
        "input_name": "input",
        "input_shape": (1, 3, 200, 200), 
        }
    },
]
