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

encoder_model_dir = Path("vae_encoder/1")
encoder_model_dir.mkdir(exist_ok=True, parents=True)
import gc
import torch

VAE_ENCODER_ONNX_PATH = encoder_model_dir / 'vae_encoder.onnx'


def convert_vae_encoder_onnx(vae: torch.nn.Module, onnx_path: Path, width:int = 512, height:int = 512):
    """
    Convert VAE model to ONNX, then IR format. 
    Function accepts pipeline, creates wrapper class for export only necessary for inference part, 
    prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        vae (torch.nn.Module): VAE PyTorch model
        onnx_path (Path): File for storing onnx model
        width (int, optional, 512): input width
        height (int, optional, 512): input height
    Returns:
        None
    """
    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, image):
            h = self.vae.encoder(image)
            moments = self.vae.quant_conv(h)
            return moments

    if not onnx_path.exists():
        vae_encoder = VAEEncoderWrapper(vae)
        vae_encoder.eval()
        image = torch.zeros((1, 3, width, height))
        with torch.no_grad():
            torch.onnx.export(vae_encoder, image, onnx_path, input_names=[
                              'init_image'], output_names=['image_latent'])
        print('VAE encoder successfully converted to ONNX')


convert_vae_encoder_onnx(vae, VAE_ENCODER_ONNX_PATH, 768, 768)

decoder_model_dir = Path("vae_decoder/1")
decoder_model_dir.mkdir(exist_ok=True, parents=True)

VAE_DECODER_ONNX_PATH = decoder_model_dir / 'vae_decoder.onnx'
VAE_DECODER_OV_PATH = VAE_DECODER_ONNX_PATH.with_suffix('.xml')


def convert_vae_decoder_onnx(vae: torch.nn.Module, onnx_path: Path, width:int = 64, height:int = 64):
    """
    Convert VAE model to ONNX, then IR format. 
    Function accepts pipeline, creates wrapper class for export only necessary for inference part, 
    prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        vae: 
        onnx_path (Path): File for storing onnx model
        width (int, optional, 64): input width
        height (int, optional, 64): input height
    Returns:
        None
    """
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            latents = 1 / 0.18215 * latents 
            return self.vae.decode(latents)

    if not onnx_path.exists():
        vae_decoder = VAEDecoderWrapper(vae)
        latents = torch.zeros((1, 4, width, height))

        vae_decoder.eval()
        with torch.no_grad():
            torch.onnx.export(vae_decoder, latents, onnx_path, input_names=[
                              'latents'], output_names=['sample'])
        print('VAE decoder successfully converted to ONNX')


convert_vae_decoder_onnx(vae, VAE_DECODER_ONNX_PATH, 96, 96)

del vae
gc.collect();
