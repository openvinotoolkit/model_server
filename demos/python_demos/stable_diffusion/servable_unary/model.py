#*****************************************************************************
# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*****************************************************************************

import io
from PIL import Image
from pyovms import Tensor
from optimum.intel.openvino import OVStableDiffusionPipeline
from diffusers import DDIMScheduler
import time
from transformers import AutoConfig

MODEL_PATH = "/model"  # relative to container

class OvmsPythonModel:
    def initialize(self, kwargs: dict):
        print("-------- Running initialize")
        self.pipe = OVStableDiffusionPipeline.from_pretrained(MODEL_PATH)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        print("-------- Model loaded")
        return True

    def execute(self, inputs: list):
        print("Running execute")
        text = bytes(inputs[0]).decode()
        image = self.pipe(text).images[0]
        output = io.BytesIO()
        image.save(output, format='PNG')
        return [Tensor("image", output.getvalue())]

