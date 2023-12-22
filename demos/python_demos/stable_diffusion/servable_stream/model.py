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
                print('callback executed ----', step, timestep, latents.shape, type(latents), np.max(latents),np.min(latents) )
                latents = 1 / 0.18215 * latents
                image = np.concatenate(
                    [self.pipe.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
                )
                pil_images = self.pipe.image_processor.postprocess(image, output_type='pil', do_denormalize=[True])
                pil_image = pil_images[0]
                output = io.BytesIO()
                pil_image.save(output, format='PNG')
                q.put((output.getvalue(),False))
                print('end callback')

            print('generating for prompt:', text)
            image = self.pipe(
                text,
                num_inference_steps=50,
                callback=callback_on_step_end_impl,
                callback_steps=2
            ).images[0]
            output = io.BytesIO()
            image.save(output, format='PNG')
            q.put((output.getvalue(),True))


        t1 = threading.Thread(target=generate)
        t1.start()
        pipeline_finished = False
        while not pipeline_finished:
            print('waiting for data...')
            image_data, pipeline_finished = q.get()
            print('got it! will serialize...')
            yield [Tensor("image", image_data)]
        yield [Tensor("end_signal", "".encode())]

