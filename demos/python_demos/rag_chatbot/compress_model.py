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

from optimum.intel import OVQuantizer, OVWeightQuantizationConfig
from optimum.intel.openvino import OVModelForCausalLM
import openvino as ov
from transformers import AutoConfig, AutoTokenizer
from pathlib import Path
import shutil
import torch
import logging
import nncf
import gc
from converter import converters, register_configs
from servable_stream.config import SUPPORTED_LLM_MODELS
import argparse

register_configs()
nncf.set_log_level(logging.ERROR)

parser = argparse.ArgumentParser(description='Script to compress LLM model based on https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot')

supported_models_list = []
for model in SUPPORTED_LLM_MODELS :
    supported_models_list.append(model)

parser.add_argument('--model',
                    required=True,
                    choices=supported_models_list,
                    help='Select the LLM model out of supported list')
args = vars(parser.parse_args())

SELECTED_MODEL = args['model']

llm_model_configuration = SUPPORTED_LLM_MODELS[SELECTED_MODEL]
MODEL_PATH = "./" + SELECTED_MODEL

print(llm_model_configuration)
pt_model_id = llm_model_configuration["model_id"]
pt_model_name = SELECTED_MODEL.split("-")[0]
model_type = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True).model_type
fp16_model_dir = Path(SELECTED_MODEL + "_FP16")
int8_model_dir = Path(SELECTED_MODEL + "_INT8_compressed_weights")
int4_model_dir = Path(SELECTED_MODEL + "_INT4_compressed_weights")

def save_tokenizer(model_id, PATH):
    print(f'Downloading tokenizer to {PATH} ...')
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f'Saving tokenizer to {PATH} ...')
    tok.save_pretrained(PATH)

def convert_to_fp16():
    if (fp16_model_dir / "openvino_model.xml").exists():
        return
    if not llm_model_configuration["remote"]:
        remote_code = llm_model_configuration.get("remote_code", False)
        model_kwargs = {}
        if remote_code:
            model_kwargs = {
                "trust_remote_code": True,
                "config": AutoConfig.from_pretrained(pt_model_id, trust_remote_code=True)
            }
        ov_model = OVModelForCausalLM.from_pretrained(
            pt_model_id, export=True, compile=False, load_in_8bit=False, **model_kwargs
        )
        ov_model.half()
        ov_model.save_pretrained(fp16_model_dir)
        del ov_model
    else:
        model_kwargs = {}
        if "revision" in llm_model_configuration:
            model_kwargs["revision"] = llm_model_configuration["revision"]
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_configuration["model_id"],
            torch_dtype=torch.float32,
            trust_remote_code=True,
            **model_kwargs
        )
        converters[pt_model_name](model, fp16_model_dir)
        del model
    gc.collect()


def convert_to_int8():
    if (int8_model_dir / "openvino_model.xml").exists():
        return
    int8_model_dir.mkdir(parents=True, exist_ok=True)
    if not llm_model_configuration["remote"]:
        remote_code = llm_model_configuration.get("remote_code", False)
        model_kwargs = {}
        if remote_code:
            model_kwargs = {
                "trust_remote_code": True,
                "config": AutoConfig.from_pretrained(pt_model_id, trust_remote_code=True)
            }
        ov_model = OVModelForCausalLM.from_pretrained(
            pt_model_id, export=True, compile=False, load_in_8bit=True, **model_kwargs
        )
        ov_model.save_pretrained(int8_model_dir)
        del ov_model
    else:
        convert_to_fp16()
        ov_model = ov.Core().read_model(fp16_model_dir / "openvino_model.xml")
        shutil.copy(fp16_model_dir / "config.json", int8_model_dir / "config.json")
        configuration_file = fp16_model_dir / f"configuration_{model_type}.py"
        if configuration_file.exists():
            shutil.copy(
                configuration_file, int8_model_dir / f"configuration_{model_type}.py"
            )
        compressed_model = nncf.compress_weights(ov_model)
        ov.save_model(compressed_model, int8_model_dir / "openvino_model.xml")
        del ov_model
        del compressed_model
    gc.collect()


def convert_to_int4():
    compression_configs = {
        "zephyr-7b-beta": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "mistral-7b": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "minicpm-2b-dpo": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "notus-7b-v1": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "neural-chat-7b-v3-1": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "llama-2-chat-7b": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "chatglm2-6b": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.72,
        },
        "qwen-7b-chat": {
            "sym": True, 
            "group_size": 128, 
            "ratio": 0.6
        },
        'red-pajama-3b-chat': {
            "sym": False,
            "group_size": 128,
            "ratio": 0.5,
        },
        "default": {
            "sym": False,
            "group_size": 128,
            "ratio": 0.8,
        },
    }

    model_compression_params = compression_configs.get(
        SELECTED_MODEL, compression_configs["default"]
    )
    if (int4_model_dir / "openvino_model.xml").exists():
        return
    int4_model_dir.mkdir(parents=True, exist_ok=True)
    if not llm_model_configuration["remote"]:
        remote_code = llm_model_configuration.get("remote_code", False)
        model_kwargs = {}
        if remote_code:
            model_kwargs = {
                "trust_remote_code" : True,
                "config": AutoConfig.from_pretrained(pt_model_id, trust_remote_code=True)
            }
        ov_model = OVModelForCausalLM.from_pretrained(
            pt_model_id, export=True, compile=False,
            quantization_config=OVWeightQuantizationConfig(bits=4, **model_compression_params),
            **model_kwargs
        )
        ov_model.save_pretrained(int4_model_dir)
        del ov_model

    else:
        convert_to_fp16()
        ov_model = ov.Core().read_model(fp16_model_dir / "openvino_model.xml")
        shutil.copy(fp16_model_dir / "config.json", int4_model_dir / "config.json")
        configuration_file = fp16_model_dir / f"configuration_{model_type}.py"
        if configuration_file.exists():
            shutil.copy(
                configuration_file, int4_model_dir / f"configuration_{model_type}.py"
            )
        mode = nncf.CompressWeightsMode.INT4_SYM if model_compression_params["sym"] else \
            nncf.CompressWeightsMode.INT4_ASYM
        del model_compression_params["sym"]
        compressed_model = nncf.compress_weights(ov_model, mode=mode, **model_compression_params)
        ov.save_model(compressed_model, int4_model_dir / "openvino_model.xml")
        del ov_model
        del compressed_model
    gc.collect()


convert_to_fp16()
convert_to_int8()
convert_to_int4()

fp16_weights = fp16_model_dir / "openvino_model.bin"
int8_weights = int8_model_dir / "openvino_model.bin"
int4_weights = int4_model_dir / "openvino_model.bin"

if fp16_weights.exists():
    print(f"Size of FP16 model is {fp16_weights.stat().st_size / 1024 / 1024:.2f} MB")
for precision, compressed_weights in zip([8, 4], [int8_weights, int4_weights]):
    if compressed_weights.exists():
        print(
            f"Size of model with INT{precision} compressed weights is {compressed_weights.stat().st_size / 1024 / 1024:.2f} MB"
        )
    if compressed_weights.exists() and fp16_weights.exists():
        print(
            f"Compression rate for INT{precision} model: {fp16_weights.stat().st_size / compressed_weights.stat().st_size:.3f}"
        )
