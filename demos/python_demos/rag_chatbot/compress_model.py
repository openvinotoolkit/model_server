#*****************************************************************************
# Copyright 2024 Intel Corporation
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

from optimum.intel import OVQuantizer
from optimum.intel.openvino import OVModelForCausalLM
import openvino as ov
from transformers import AutoConfig, AutoTokenizer
from pathlib import Path
import shutil
import torch
import logging
import nncf
import gc
from converter import converters
from servable_stream.config import SUPPORTED_LLM_MODELS
import argparse
nncf.set_log_level(logging.ERROR)

parser = argparse.ArgumentParser(description='Script to compress LLM model based on https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot')

supported_models_list = [model_name for model in SUPPORTED_LLM_MODELS.values() for model_name in model]

parser.add_argument('--model',
                    required=True,
                    choices=supported_models_list,
                    help='Select the LLM model out of supported list')
args = vars(parser.parse_args())

SELECTED_MODEL = args['model']
LANGUAGE = 'English'

model_configuration = SUPPORTED_LLM_MODELS[LANGUAGE][SELECTED_MODEL]
MODEL_PATH = "./servable_stream/" + SELECTED_MODEL

print(model_configuration)
pt_model_id = model_configuration["model_id"]
pt_model_name = SELECTED_MODEL.split("-")[0]
model_type = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True).model_type
fp16_model_dir = Path("./servable_stream/" + SELECTED_MODEL + "_FP16")
int8_model_dir = Path("./servable_stream/" + SELECTED_MODEL + "_INT8_compressed_weights")
int4_model_dir = Path("./servable_stream/" + SELECTED_MODEL + "_INT4_compressed_weights")

def save_tokenizer(model_id, PATH):
    print(f'Downloading tokenizer to {PATH} ...')
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f'Saving tokenizer to {PATH} ...')
    tok.save_pretrained(PATH)

def convert_to_fp16():
    if (fp16_model_dir / "openvino_model.xml").exists():
        return
    if not model_configuration["remote"]:
        ov_model = OVModelForCausalLM.from_pretrained(
            pt_model_id, export=True, compile=False, load_in_8bit=False
        )
        ov_model.half()
        ov_model.save_pretrained(fp16_model_dir)
        del ov_model
    else:
        model_kwargs = {}
        if "revision" in model_configuration:
            model_kwargs["revision"] = model_configuration["revision"]
        model = AutoModelForCausalLM.from_pretrained(
            model_configuration["model_id"],
            torch_dtype=torch.float32,
            trust_remote_code=True,
            **model_kwargs
        )
        converters[pt_model_name](model, fp16_model_dir)
        del model
    gc.collect()
    save_tokenizer(pt_model_id, fp16_model_dir)


def convert_to_int8():
    if (int8_model_dir / "openvino_model.xml").exists():
        return
    int8_model_dir.mkdir(parents=True, exist_ok=True)
    if not model_configuration["remote"]:
        if fp16_model_dir.exists():
            ov_model = OVModelForCausalLM.from_pretrained(fp16_model_dir, compile=False)
        else:
            ov_model = OVModelForCausalLM.from_pretrained(
                pt_model_id, export=True, compile=False, load_in_8bit=False
            )
            ov_model.half()
        quantizer = OVQuantizer.from_pretrained(ov_model)
        quantizer.quantize(save_directory=int8_model_dir, weights_only=True)
        del quantizer
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
    save_tokenizer(pt_model_id, int8_model_dir)


def convert_to_int4():
    compression_configs = {
        "zephyr-7b-beta": {
            "mode": nncf.CompressWeightsMode.INT4_SYM,
            "group_size": 64,
            "ratio": 0.6,
        },
        "mistral-7b": {
            "mode": nncf.CompressWeightsMode.INT4_SYM,
            "group_size": 64,
            "ratio": 0.6,
        },
        "notus-7b-v1": {
            "mode": nncf.CompressWeightsMode.INT4_SYM,
            "group_size": 64,
            "ratio": 0.6,
        },
        "neural-chat-7b-v3-1": {
            "mode": nncf.CompressWeightsMode.INT4_SYM,
            "group_size": 64,
            "ratio": 0.6,
        },
        "llama-2-chat-7b": {
            "mode": nncf.CompressWeightsMode.INT4_SYM,
            "group_size": 128,
            "ratio": 0.8,
        },
        "chatglm2-6b": {
            "mode": nncf.CompressWeightsMode.INT4_SYM,
            "group_size": 128,
            "ratio": 0.72,
            "ignored_scope": nncf.IgnoredScope(["__module.transformer/aten::index_67/Gather"])
        },
        "qwen-7b-chat": {
            "mode": nncf.CompressWeightsMode.INT4_SYM,
            "group_size": 128,
            "ratio": 0.6
        },
        'red-pajama-3b-chat': {
            "mode": nncf.CompressWeightsMode.INT4_ASYM,
            "group_size": 128,
            "ratio": 0.5,
        },
        "default": {
            "mode": nncf.CompressWeightsMode.INT4_ASYM,
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
    if not model_configuration["remote"]:
        if not fp16_model_dir.exists():
            model = OVModelForCausalLM.from_pretrained(
                pt_model_id, export=True, compile=False, load_in_8bit=False
            ).half()
            model.config.save_pretrained(int4_model_dir)
            ov_model = model._original_model
            del model
            gc.collect()
        else:
            ov_model = ov.Core().read_model(fp16_model_dir / "openvino_model.xml")
            shutil.copy(fp16_model_dir / "config.json", int4_model_dir / "config.json")

    else:
        convert_to_fp16()
        ov_model = ov.Core().read_model(fp16_model_dir / "openvino_model.xml")
        shutil.copy(fp16_model_dir / "config.json", int4_model_dir / "config.json")
        configuration_file = fp16_model_dir / f"configuration_{model_type}.py"
        if configuration_file.exists():
            shutil.copy(
                configuration_file, int4_model_dir / f"configuration_{model_type}.py"
            )
    compressed_model = nncf.compress_weights(ov_model, **model_compression_params)
    ov.save_model(compressed_model, int4_model_dir / "openvino_model.xml")
    del ov_model
    del compressed_model
    gc.collect()
    save_tokenizer(pt_model_id, int4_model_dir)


convert_to_fp16()
convert_to_int8()
convert_to_int4()
