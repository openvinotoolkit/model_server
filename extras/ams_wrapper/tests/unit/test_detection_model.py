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

import json
from typing import Dict

import pytest
import numpy as np

from src.api.models.detection_model import DetectionModel
from src.api.models.model_config import ModelOutputConfiguration

MOCK_INFERENCE_OUTPUT = {
    'result':
    np.array([
        [
            [
                # id, label, confidence, x_min, y_min, x_max, y_max
                [0, 1.0, 0.97, 0.17, 0.15, 0.5, 0.7],
                [1, 1.0, 0.46, 0.12, 0.11, 0.6, 0.5],
            ]
        ]
    ])
}


@pytest.fixture
def fake_output_config() -> Dict[str, ModelOutputConfiguration]:
    return {
        'result': ModelOutputConfiguration(output_name='result',
                                           value_index_mapping={
                                               "image_id": 0,
                                               "value": 1,
                                               "confidence": 2,
                                               "x_min": 3,
                                               "y_min": 4,
                                               "x_max": 5,
                                               "y_max": 6
                                           },
                                           classes={
                                               "background": 0.0,
                                               "vehicle": 1.0
                                           }
                                           )
    }


@pytest.mark.parametrize("inference_output,expected_response", [
                        (MOCK_INFERENCE_OUTPUT,
                         {
                          "inferences": [
                            {
                              "type": "entity",
                              "subtype": None,
                              "entity": {
                                "tag": {"value": "vehicle", "confidence": 0.97},
                                "box": {"l": 0.17, "t": 0.15, "w": 0.32999999999999996, "h": 0.5499999999999999}
                                }
                            },
                            {
                              "type": "entity",
                              "subtype": None,
                              "entity": {
                                "tag": {"value": "vehicle", "confidence": 0.46},
                                "box": {"l": 0.12, "t": 0.11, "w": 0.48, "h": 0.39}
                              }
                            }
                          ]
                         }
                         )])
def test_postprocess_inference_output(inference_output, expected_response, fake_output_config):
    model = DetectionModel(endpoint=None, ovms_connector=None, input_configs=None,
                           output_configs=fake_output_config)
    assert model.postprocess_inference_output(
        inference_output) == json.dumps(expected_response)


@pytest.mark.parametrize("inference_output,expected_response,top_k", [
                        (MOCK_INFERENCE_OUTPUT,
                         {
                          "inferences": [
                            {
                              "type": "entity",
                              "subtype": None,
                              "entity": {
                                "tag": {"value": "vehicle", "confidence": 0.97},
                                "box": {"l": 0.17, "t": 0.15, "w": 0.32999999999999996, "h": 0.5499999999999999}
                                }
                            }
                          ]
                         },
                         1
                         )])
def test_postprocess_inference_output_top_k(inference_output, expected_response, top_k, fake_output_config):
    fake_output_config['result'].top_k_results = top_k
    model = DetectionModel(endpoint=None, ovms_connector=None, input_configs=None,
                           output_configs=fake_output_config)
    assert model.postprocess_inference_output(
        inference_output) == json.dumps(expected_response)


@pytest.mark.parametrize("inference_output,expected_response,confidence_threshold", [
                        (MOCK_INFERENCE_OUTPUT,
                         {
                          "inferences": [
                            {
                              "type": "entity",
                              "subtype": None,
                              "entity": {
                                "tag": {"value": "vehicle", "confidence": 0.97},
                                "box": {"l": 0.17, "t": 0.15, "w": 0.32999999999999996, "h": 0.5499999999999999}
                                }
                            }
                          ]
                         },
                         0.5
                         )])
def test_postprocess_inference_output_confidence_threshold(inference_output, expected_response,
                                                           confidence_threshold, fake_output_config):
    fake_output_config['result'].confidence_threshold = confidence_threshold
    model = DetectionModel(endpoint=None, ovms_connector=None, input_configs=None,
                           output_configs=fake_output_config)

    assert model.postprocess_inference_output(
        inference_output) == json.dumps(expected_response)
