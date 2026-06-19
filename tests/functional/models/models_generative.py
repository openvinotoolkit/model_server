#
# Copyright (c) 2026 Intel Corporation
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

import os
import shutil

from dataclasses import dataclass
from pathlib import Path

from tests.functional.config import generative_models_local_path
from tests.functional.constants.paths import Paths
from tests.functional.models.models import ModelInfo, ModelType
from tests.functional.models.models_datasets import (
    FeatureExtractionModelDataset,
    LargeLanguageModelDataset,
    RerankModelDataset,
    SingleMessageLanguageModelDataset,
)


@dataclass
class GenerativeModel(ModelInfo):
    model_type: ModelType = ModelType.IR
    is_generative: bool = True
    is_local: bool = True
    precision: str = "INT8"
    precision_dir: str = "INT8"
    parent_name: str = None
    parent_base_dir: str = os.path.join("pytorch", "ov")
    parent_precision_dir: str = "OV_FP16-INT8_ASYM"
    max_position_embeddings: int = None
    model_path_on_parent_host: str = None
    model_subpath: str = None
    single_message_dataset: bool = False
    allows_reasoning: bool = False
    is_llm: bool = False
    is_feature_extraction: bool = False
    is_rerank: bool = False
    is_image_generation: bool = False
    is_audio: bool = False
    is_hf_direct_load: bool = False
    is_agentic: bool = False
    gguf_filename: str = None
    pooling: str = None
    transformers_v4_required: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._own_field_defaults = {}
        for name in getattr(cls, '__annotations__', {}):
            if name in cls.__dict__:
                cls._own_field_defaults[name] = cls.__dict__[name]

    def __post_init__(self):
        self.model_base_path_on_host = generative_models_local_path
        self.model_subpath = os.path.join(self.precision_dir, Path(self.name))
        self.model_path_on_host = os.path.join(self.model_base_path_on_host, self.model_subpath)
        self.base_path = os.path.join(Paths.MODELS_PATH_INTERNAL, self.model_subpath)

    def get_default_dataset(self):
        return

    def prepare_input_data(self, batch_size=None, input_key=None, dataset=None, input_data_type=None):
        if dataset is not None:
            input_data = {input_name: dataset().get_data(None, None, None) for input_name in self.input_names}
        elif input_data_type == "string":
            input_data = {
                input_name: self.inputs[input_name]["dataset"].get_string_data()
                for input_name in self.input_names
            }
        else:
            input_data = {
                input_name: self.inputs[input_name]["dataset"].get_data(None, None, None)
                for input_name in self.input_names
            }
        return input_data


@dataclass
class LargeLanguageModel(GenerativeModel):
    is_llm: bool = True

    def get_default_dataset(self):
        if self.single_message_dataset:
            return SingleMessageLanguageModelDataset
        return LargeLanguageModelDataset


@dataclass
class FeatureExtractionModel(GenerativeModel):
    use_subconfig: bool = True
    is_feature_extraction: bool = True
    pooling: str = "CLS"

    def get_default_dataset(self):
        return FeatureExtractionModelDataset


@dataclass
class RerankModel(GenerativeModel):
    is_rerank: bool = True
    use_subconfig: bool = True

    @staticmethod
    def get_default_dataset():
        return RerankModelDataset


@dataclass
class GenerativeModelHuggingFace(GenerativeModel):
    is_local: bool = False
    is_hf_direct_load: bool = True
    input_name: str = "input"
    precision: str = "INT4"
    model_timeout: int = 900

    def _apply_diamond_defaults(self):
        """Fix field defaults for diamond inheritance.

        When a class inherits from both LargeLanguageModelHuggingFace and a specialized
        type (e.g. ImageGenerationModel), LargeLanguageModelHuggingFace's inherited field
        defaults override the specialized type's directly-defined defaults. This method
        restores the correct defaults from specialized parent classes.
        """
        cls = type(self)
        seen_fields = set()
        seen_fields.update(getattr(cls, '_own_field_defaults', {}).keys())
        seen_fields.update(getattr(GenerativeModelHuggingFace, '_own_field_defaults', {}).keys())
        for base in cls.__mro__:
            if base in (cls, object, GenerativeModelHuggingFace, GenerativeModel, ModelInfo):
                continue
            own_defaults = getattr(base, '_own_field_defaults', {})
            for field_name, default_value in own_defaults.items():
                if field_name not in seen_fields:
                    setattr(self, field_name, default_value)
                    seen_fields.add(field_name)

    def __post_init__(self):
        self._apply_diamond_defaults()
        if self.is_local:
            self.model_base_path_on_host = generative_models_local_path
            self.model_path_on_host = os.path.join(self.model_base_path_on_host, Path(self.name))
            self.model_subpath = os.path.join("models_ov_hf", Path(self.name))
        self.base_path = os.path.join(Paths.MODELS_PATH_INTERNAL, self.name)
        self.set_additional_model_params()

    def prepare_resources(self, base_location):
        models_dir = Path(base_location, Paths.MODELS_PATH_NAME)
        models_dir.mkdir(exist_ok=True, parents=True)
        if self.is_local:
            models_sub_dir = Path(models_dir, self.name)
            if not models_sub_dir.exists():
                shutil.copytree(self.model_path_on_host, models_sub_dir)
        return [str(models_dir)]

    def prepare_input_data(self, batch_size=None, input_key=None, dataset=None, input_data_type=None):
        if dataset is not None:
            dataset_obj = dataset if not isinstance(dataset, type) else dataset()
        else:
            dataset_obj = self.get_default_dataset()()
        if input_data_type == "string":
            input_data = {self.input_name: dataset_obj.get_string_data()}
        else:
            input_data = {self.input_name: dataset_obj.get_data(None, None, None)}
        return input_data


@dataclass
class Qwen3Embedding06BFp16OvHf(GenerativeModelHuggingFace, FeatureExtractionModel):
    name: str = "OpenVINO/Qwen3-Embedding-0.6B-fp16-ov"
    precision: str = "FP16"
    pooling: str = "LAST"
    is_local: bool = True
