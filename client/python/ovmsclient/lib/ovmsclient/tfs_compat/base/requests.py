#
# Copyright (c) 2021 Intel Corporation
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

from abc import ABC


class PredictRequest(ABC):

    def __init__(self, inputs, model_name, model_version):
        self.inputs = inputs
        self.model_name = model_name
        self.model_version = model_version


class ModelMetadataRequest(ABC):

    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version


class ModelStatusRequest(ABC):

    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version


def _check_model_spec(model_name, model_version):

    if not isinstance(model_name, str):
        raise TypeError(f'model_name type should be string, but is {type(model_name).__name__}')

    if not isinstance(model_version, int):
        raise TypeError(f'model_version type should be int, but is {type(model_version).__name__}')

    if model_version.bit_length() > 63 or model_version < 0:
        raise ValueError(f'model_version should be in range <0, {2**63-1}>')
