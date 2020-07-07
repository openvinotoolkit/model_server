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
import io

import pytest
import unittest.mock as mock
import falcon

from src.api.models.model_builder import ModelBuilder
from src.api.models.model import Model
from src.api.ovms_connector import OvmsUnavailableError, ModelNotFoundError, \
    RequestProcessingError
from src.preprocessing import ImageResizeError, ImageDecodeError, ImageTransformError


@pytest.fixture()
def test_model_config():
    config_file_content = {
        "endpoint": "some_color_model",
        "model_type": "classification_attributes",
        "outputs": [
            {
                "output_name": "prob",
                "classes": {
                    "white": 0.0,
                    "gray": 1.0,
                    "yellow": 2.0,
                    "red": 3.0,
                    "green": 4.0,
                    "blue": 5.0,
                    "black": 6.0
                },
            },
        ],
        "inputs": [
            {
                "input_name": "result",
            }
        ],
        "ovms_mapping": {
            "model_name": "color_model",
            "model_version": 0
        }
    }
    return config_file_content


@pytest.fixture()
def test_model(tmpdir, test_model_config):
    config_file_path = tmpdir.join("model_config.json")
    with open(config_file_path, mode='w') as config_file:
        json.dump(test_model_config, config_file)
    return ModelBuilder.build_model(config_file_path, 4000)


class FakeModel(Model):
    def postprocess_inference_output(self, inference_output: dict) -> str:
        pass


exc = [ValueError, TypeError, ImageDecodeError,
       ImageResizeError, ImageTransformError]


@pytest.mark.parametrize('exceptions', exc)
def test_model_image_preprocessing_exception(mocker, test_model, exceptions):

    mod = test_model
    mod.preprocess_binary_image = mock.Mock(side_effect=exceptions)
    resp = falcon.Response()
    req = type('test', (), {})()
    req.headers = {}
    req.bounded_stream = io.BytesIO(b"")
    mod.on_post(req, resp)
    assert resp.status == falcon.HTTP_400


exceptions_and_statuses = [(ValueError, falcon.HTTP_400),
                           (TypeError, falcon.HTTP_400), (ModelNotFoundError,
                                                          falcon.HTTP_500),
                           (OvmsUnavailableError, falcon.HTTP_503), (RequestProcessingError, falcon.HTTP_500)]


@pytest.mark.parametrize("exceptions", exceptions_and_statuses)
def test_model_inference_exceptions(mocker, tmpdir, test_model, exceptions):

    mod = test_model
    mod.ovms_connector.send = mock.Mock(
            side_effect=exceptions[0])

    mod.preprocess_binary_image = mock.Mock()
    resp = falcon.Response()
    req = type('test', (), {})()
    req.headers = {}
    req.bounded_stream = io.BytesIO(b"")
    mod.on_post(req, resp)
    assert resp.status == exceptions[1]
