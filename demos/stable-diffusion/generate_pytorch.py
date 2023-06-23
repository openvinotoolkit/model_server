# Copyright (c) 2023 Intel Corporation
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
# Copyright (c) 2023 Intel Corporation
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
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to("cpu")

#from stable_diffusion import OVStableDiffusionPipeline

import numpy as np
from torch import Generator

from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler

from transformers import CLIPTokenizer

import argparse

parser = argparse.ArgumentParser(description='Client for single face analysis pipeline')
parser.add_argument('--adapter', required=False, default='openvino', choices=['openvino', 'ovms'], help='Specify inference executor adapter type')
parser.add_argument('--prompt', required=False, type=str, default="see shore at midnight, epic vista, beach", help='Text prompt')
parser.add_argument('--negative', required=False, type=str, default='frames, borderline, text, character, duplicate, error, out of frame, watermark', help='Negative prompt')
parser.add_argument('--steps', required=False, default=20, type=int, help='Number of steps')
parser.add_argument('--seed', required=False, default=None, type=int, help='Seed')
args = vars(parser.parse_args())

conf = {"num_train_timesteps": 1000, "beta_start" : 0.00085, 'beta_end': 0.012, 'beta_schedule': 'scaled_linear', 'trained_betas': None, 'skip_prk_steps': True, 'set_alpha_to_one': False, 'prediction_type': 'epsilon', 'steps_offset': 1, '_class_name': 'PNDMScheduler', '_diffusers_version': '0.10.0.dev0', 'clip_sample': False}

print("Initializing pipeline")
#scheduler = LMSDiscreteScheduler.from_config(conf)
#tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to("cpu")
print("Pipeline initialized")
prompt = args["prompt"]

negative = args["negative"]

seed = args["seed"]
steps= args["steps"]

print('Pipeline settings')
print(f'Input text: {prompt}')
print(f'Negative text: {negative}')
print(f'Seed: {seed}')
print(f'Number of steps: {steps}')


result = pipe(prompt, negative_prompt=negative, num_inference_steps=steps, height=768,width=768, generator=Generator().manual_seed(seed))

final_image = result.images[0]
final_image.save('result.png')
