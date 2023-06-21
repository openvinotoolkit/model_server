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

sd2_1_model_dir = Path("text_encoder/1")
sd2_1_model_dir.mkdir(parents=True,exist_ok=True)

import gc
import torch

TEXT_ENCODER_ONNX_PATH = sd2_1_model_dir / 'text_encoder.onnx'


def convert_encoder_onnx(text_encoder: torch.nn.Module, onnx_path:Path):
    """
    Convert Text Encoder model to ONNX.
    Function accepts pipeline, prepares example inputs for ONNX conversion via torch.export,
    Parameters:
        text_encoder (torch.nn.Module): text encoder PyTorch model
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """
    if not onnx_path.exists():
        input_ids = torch.ones((1, 77), dtype=torch.long)
        # switch model to inference mode
        text_encoder.eval()

        # disable gradients calculation for reducing memory consumption
        with torch.no_grad():
            # export model to ONNX format
            torch.onnx._export(
                text_encoder,  # model instance
                input_ids,  # inputs for model tracing
                onnx_path,  # output file for saving result
                input_names=['tokens'],  # model input name for onnx representation
                output_names=['last_hidden_state', 'pooler_out'],  # model output names for onnx representation
                opset_version=14,  # onnx opset version for export,
                onnx_shape_inference=False
            )
        print('Text Encoder successfully converted to ONNX')


convert_encoder_onnx(text_encoder, TEXT_ENCODER_ONNX_PATH)

del text_encoder
gc.collect()
