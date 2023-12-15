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
import torch
import numpy as np
import threading
from queue import Queue

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

        q = Queue()
        def generate():
            def callback_on_step_end_impl(step, timestep,
                    latents):
                print('callback executed ----', step, timestep, latents.shape)

                latents = 1 / 0.18215 * latents
                with torch.no_grad():
                    image = self.pipe.vae_decoder(latents)

                image = torch.from_numpy(image[0])
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                image = (image * 255).round().astype("uint8")
                images = (image * 255).round().astype("uint8")
                pil_images = [Image.fromarray(image) for image in images]
                pil_image = pil_images[0]

                output = io.BytesIO()
                pil_image.save(output, format='PNG')
                q.put(output.getvalue())
                print('end callback')
            
            print('inferencing:', text)
            image = self.pipe(
                text,
                num_inference_steps=50,
                callback=callback_on_step_end_impl
            ).images[0]
            output = io.BytesIO()
            image.save(output, format='PNG')
            q.put(output.getvalue())


        t1 = threading.Thread(target=generate)
        t1.start()

        for i in range(51):
            print('waiting for data...', i)
            my_data = q.get()
            print('got it! will serialize...')
            yield [Tensor("OUTPUT", my_data)]
        yield [Tensor("END_SIGNAL", "".encode())]
