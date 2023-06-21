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

# for reducing memory consumption get all components from pipeline independently
text_encoder = pipe.text_encoder
text_encoder.eval()
unet = pipe.unet
unet.eval()
vae = pipe.vae
vae.eval()

conf = pipe.scheduler.config

del pipe

from pathlib import Path

sd2_1_model_dir = Path("unet/1")
sd2_1_model_dir.mkdir(exist_ok=True, parents=True)

import gc
import torch
import numpy as np

UNET_ONNX_PATH = sd2_1_model_dir / 'unet.onnx'


def convert_unet_onnx(unet:torch.nn.Module, onnx_path:Path, num_channels:int = 4, width:int = 64, height:int = 64):
    """
    Convert Unet model to ONNX, then IR format. 
    Function accepts pipeline, prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        unet (torch.nn.Module): UNet PyTorch model
        onnx_path (Path): File for storing onnx model
        num_channels (int, optional, 4): number of input channels
        width (int, optional, 64): input width
        height (int, optional, 64): input height
    Returns:
        None
    """
    if not onnx_path.exists():
        # prepare inputs
        
        encoder_hidden_state = torch.ones((2, 77, 1024))
        latents_shape = (2, num_channels, width, height)

        latents = torch.randn(latents_shape)
        t = torch.from_numpy(np.array(1, dtype=np.float32))
        print("tshape:",np.array(1, dtype=np.float32).shape)

        # model size > 2Gb, it will be represented as onnx with external data files, we will store it in separated directory for avoid a lot of files in current directory
        onnx_path.parent.mkdir(exist_ok=True, parents=True)
        unet.eval()

        with torch.no_grad():
            torch.onnx._export(
                unet, 
                (latents, t, encoder_hidden_state), str(onnx_path),
                input_names=['latent_model_input', 't', 'encoder_hidden_states'],
                output_names=['out_sample'],
                onnx_shape_inference=False
            )
        print('U-Net successfully converted to ONNX')


convert_unet_onnx(unet, UNET_ONNX_PATH, width=96, height=96)
del unet
gc.collect()
