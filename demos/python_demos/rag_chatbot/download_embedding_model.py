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

import argparse
from pathlib import Path

from servable_stream.config import SUPPORTED_EMBEDDING_MODELS, SUPPORTED_LLM_MODELS
from converter import converters

from transformers import (
    AutoModel,
    AutoTokenizer,
)

parser = argparse.ArgumentParser(description='Script to download LLM model based on https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot')

supported_models_list = []
for key, _ in SUPPORTED_EMBEDDING_MODELS.items() :
    supported_models_list.append(key)

parser.add_argument('--model',
                    required=True,
                    choices=supported_models_list,
                    help='Select the LLM model out of supported list')
args = parser.parse_args()

SELECTED_MODEL = args.model

model_configuration = SUPPORTED_EMBEDDING_MODELS[SELECTED_MODEL]

MODEL_PATH = "./" + SELECTED_MODEL

embedding_model_dir = Path(SELECTED_MODEL)
embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS[SELECTED_MODEL]

if not (embedding_model_dir / "openvino_model.xml").exists():
    model = AutoModel.from_pretrained(embedding_model_configuration["model_id"])
    converters[SELECTED_MODEL](model, embedding_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_configuration["model_id"])
    tokenizer.save_pretrained(embedding_model_dir)
    del model
