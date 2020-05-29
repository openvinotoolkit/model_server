#
# Copyright (c) 2020 Intel Corporation
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

import pytest

from src.api.models.model_builder import ModelBuilder
from src.api.models.model_config import ValidationError
from unittest.mock import mock_open


@pytest.mark.parametrize("exception, reraised_exception", [
    (FileNotFoundError, FileNotFoundError),
    (Exception, ValueError)
    ])
def test_load_model_config_bad_file(mocker, exception, reraised_exception):
    open_mock = mocker.patch("src.api.models.model_builder.open")
    open_mock.side_effect = exception
    with pytest.raises(reraised_exception):
        ModelBuilder.build_model("path", 4000)


VALID_CONFIG = {
    "endpoint": "ageGenderRecognition",
    "model_type": "classification_attributes",
    "inputs": [
        {
            "input_name": "data",
            "input_format": "NCHW",
            "color_format": "BGR",
            "target_height": 62,
            "target_width": 62
        }
    ],
    "outputs": [
        {
            "output_name": "prob",
            "classes": {
                "female": 0.0,
                "male": 1.0
            },
            "is_softmax": True
        },
        {
            "output_name": "age_conv3",
            "classes": {
                "age": 0.0
            },
            "is_softmax": False,
            "value_multiplier": 100.0
        }
    ],
    "ovms_mapping": {
        "model_name": "age_gender_recognition",
        "model_version": 0
    }
}


def modified_dict(original_dict, key, new_value):
    new_dict = dict(original_dict)
    new_dict[key] = new_value
    return new_dict


INVALID_CONFIGS = [
    {"random": "dict"},
    {key: VALID_CONFIG[key] for key in VALID_CONFIG if key != "endpoint"},
    {key: VALID_CONFIG[key] for key in VALID_CONFIG if key != "model_type"},
    {key: VALID_CONFIG[key] for key in VALID_CONFIG if key != "inputs"},
    {key: VALID_CONFIG[key] for key in VALID_CONFIG if key != "outputs"},
    {key: VALID_CONFIG[key] for key in VALID_CONFIG if key != "ovms_mapping"},
    modified_dict(VALID_CONFIG, "endpoint", 1),
    modified_dict(VALID_CONFIG, "model_type", ["detection", "classification"]),
    modified_dict(VALID_CONFIG, "inputs", [1, 2, 3]),
    modified_dict(VALID_CONFIG, "outputs", {"output_name": "output"}),
    modified_dict(VALID_CONFIG, "ovms_mapping", "model_name")
]


@pytest.mark.parametrize("invalid_config", INVALID_CONFIGS)
def test_load_model_config_invalid(mocker, invalid_config):
    mocker.patch("src.api.models.model_builder.open", mock_open())
    json_load_mock = mocker.patch("src.api.models.model_builder.json.load")
    json_load_mock.return_value = invalid_config

    with pytest.raises(ValidationError):
        ModelBuilder.build_model("path", 4000)


INVALID_INPUTS = [
    {key: VALID_CONFIG["inputs"][0][key] for key in VALID_CONFIG["inputs"][0] if key != "input_name"},
    modified_dict(VALID_CONFIG["inputs"][0], "input_name", 1),
    modified_dict(VALID_CONFIG["inputs"][0], "channels", "string"),
    modified_dict(VALID_CONFIG["inputs"][0], "target_height", "string"),
    modified_dict(VALID_CONFIG["inputs"][0], "target_width", [32, 5]),
    modified_dict(VALID_CONFIG["inputs"][0], "color_format", "BRG"),
    modified_dict(VALID_CONFIG["inputs"][0], "scale", -3.0),
    modified_dict(VALID_CONFIG["inputs"][0], "standardization", 123),
    modified_dict(VALID_CONFIG["inputs"][0], "input_format", "NWHC"),
    modified_dict(VALID_CONFIG["inputs"][0], "additions", ["add1", "add2"])
]


@pytest.mark.parametrize("invalid_inputs", INVALID_INPUTS)
def test_load_input_configs_invalid(mocker, invalid_inputs):
    model_config = modified_dict(VALID_CONFIG, "inputs", [invalid_inputs])
    with pytest.raises(ValidationError):
        ModelBuilder._load_input_configs(model_config)


INVALID_OUTPUTS = [
    {key: VALID_CONFIG["outputs"][0][key] for key in VALID_CONFIG["outputs"][0] if key != "output_name"},
    modified_dict(VALID_CONFIG["outputs"][0], "output_name", 1),
    modified_dict(VALID_CONFIG["outputs"][0], "is_softmax", "string"),
    modified_dict(VALID_CONFIG["outputs"][0], "value_multiplier", "string"),
    modified_dict(VALID_CONFIG["outputs"][0], "value_index_mapping", {"1": "string"}),
    modified_dict(VALID_CONFIG["outputs"][0], "value_index_mapping", {1: "string"}),
    modified_dict(VALID_CONFIG["outputs"][0], "value_index_mapping", {1: 1}),
    modified_dict(VALID_CONFIG["outputs"][0], "classes", {"1": "string"}),
    modified_dict(VALID_CONFIG["outputs"][0], "classes", {1: "string"}),
    modified_dict(VALID_CONFIG["outputs"][0], "classes", {1: 1}),
    modified_dict(VALID_CONFIG["outputs"][0], "confidence_threshold", 1.1),
    modified_dict(VALID_CONFIG["outputs"][0], "top_k_results", 0),
    modified_dict(VALID_CONFIG["outputs"][0], "additions", ["add1", "add2"])
]


@pytest.mark.parametrize("invalid_outputs", INVALID_OUTPUTS)
def test_load_outputs_configs_invalid(mocker, invalid_outputs):
    model_config = modified_dict(VALID_CONFIG, "outputs", [invalid_outputs])
    with pytest.raises(ValidationError):
        ModelBuilder._load_output_configs(model_config)
