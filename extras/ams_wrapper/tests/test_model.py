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

from src.api.models.model import Model
from src.api.models.model_config import ValidationError
from src.api.ovms_connector import OvmsUnavailableError, ModelNotFoundError, \
        OvmsConnector, RequestProcessingError
from src.preprocessing import ImageResizeError, ImageDecodeError, ImageTransformError


@pytest.fixture()
def test_model_config():
    config_file_content = {
        "model_name": "age-gender-recognition-retail-0013",
        "outputs": [
            {
                "output_name": "prob",
            },
        ],
        "inputs": [
            {
                "input_name": "result",
            }
        ]
    }
    return config_file_content


@pytest.mark.parametrize("input_name", ["result"])
@pytest.mark.parametrize("channels", [None, 3])
@pytest.mark.parametrize("target_size", [(None, None), (256, 256)])
@pytest.mark.parametrize("input_format", [None, 'NCHW', 'NHWC'])
@pytest.mark.parametrize("scale", [None, 1/255])
@pytest.mark.parametrize("standardization", [None, True, False])
@pytest.mark.parametrize("color_format", [None, 'RGB', 'BGR'])
def test_model_load_valid_input_config(tmpdir, test_model_config,
                                       input_name, channels, target_size,
                                       color_format, scale, standardization,
                                       input_format):
    for input_param, input_param_value in [('channels', channels),
                                           ('target_height', target_size[0]),
                                           ('target_width', target_size[1]),
                                           ('input_format', input_format),
                                           ('scale', scale),
                                           ('standardization', standardization),
                                           ('color_format', color_format)]:
        if input_param_value is not None:
            test_model_config['inputs'][0][input_param] = input_param_value

    config_file_path = tmpdir.join("model_config.json")
    with open(config_file_path, mode='w') as config_file:
        json.dump(test_model_config, config_file)

    Model.load_input_configs(config_file_path)


@pytest.mark.parametrize("input_name", ["result"])
@pytest.mark.parametrize("channels", [None, 'two'])
@pytest.mark.parametrize("target_size", [(None, 256), (256, None)])
@pytest.mark.parametrize("input_format", ['CHWN'])
@pytest.mark.parametrize("scale", [0, 'zero'])
@pytest.mark.parametrize("standardization", ['no'])
@pytest.mark.parametrize("color_format", ['RGR'])
def test_model_load_invalid_input_config(tmpdir, test_model_config,
                                         input_name, channels, target_size,
                                         input_format, scale, standardization,
                                         color_format):
    for input_param, input_param_value in [('channels', channels),
                                           ('target_height', target_size[0]),
                                           ('target_width', target_size[1]),
                                           ('input_format', input_format),
                                           ('scale', scale),
                                           ('standardization', standardization),
                                           ('color_format', color_format)]:
        if input_param_value is not None:
            test_model_config['inputs'][0][input_param] = input_param_value

    config_file_path = tmpdir.join("model_config.json")
    with open(config_file_path, mode='w') as config_file:
        json.dump(test_model_config, config_file)

    with pytest.raises(ValidationError):
        Model.load_input_configs(config_file_path)


def test_model_load_non_existing_config():
    with pytest.raises(ValueError):
        Model.load_model_config('/not-existing-path')


@pytest.mark.parametrize("output_name", ["prob"])
@pytest.mark.parametrize("value_index_mapping", [None, {"male": 0.0, "female": 1.0}, {"male": 1, "female": 1}])
@pytest.mark.parametrize("classes", [None, {"red": 0.0, "green": 1.0}])
def test_model_load_valid_output_config(tmpdir, test_model_config,
                                       output_name, value_index_mapping,
                                       classes):
    for param, param_value in [('output_name', output_name),
                                ('value_index_mapping', value_index_mapping),
                                ('classes', classes)]:
        if param_value is not None:
            test_model_config['outputs'][0][param] = param_value

    config_file_path = tmpdir.join("model_config.json")
    with open(config_file_path, mode='w') as config_file:
        json.dump(test_model_config, config_file)

    Model.load_output_configs(config_file_path)

@pytest.mark.parametrize("output_name", ["prob"])
@pytest.mark.parametrize("value_index_mapping", [{"male": '0.0', "female": '1.0'}, {}])
@pytest.mark.parametrize("classes", [{}, {1: 'green', 0: 'red'}])
@pytest.mark.parametrize("confidence_threshold", ['zero', 120])
@pytest.mark.parametrize("top_k_results", ['zero', -1])
def test_model_load_invalid_output_config(tmpdir, test_model_config,
                                         output_name, value_index_mapping,
                                         classes, confidence_threshold, top_k_results):
    for param, param_value in [('output_name', output_name),
                               ('value_index_mapping', value_index_mapping),
                               ('classes', classes),
                               ('confidence_threshold', confidence_threshold),
                               ('top_k_results', top_k_results)]:
        if param_value is not None:
            test_model_config['outputs'][0][param] = param_value

    config_file_path = tmpdir.join("model_config.json")
    with open(config_file_path, mode='w') as config_file:
        json.dump(test_model_config, config_file)

    with pytest.raises(ValidationError):
        config = Model.load_output_configs(config_file_path)

class TestModel(Model):
    def postprocess_inference_output(self, inference_output: dict) -> str:
        pass

exc = [ValueError, TypeError, ImageDecodeError, ImageResizeError, ImageTransformError]
@pytest.mark.parametrize('exceptions', exc)
@mock.patch('src.api.models.model.Model.load_model_config')
@mock.patch('src.api.models.model.Model.load_input_configs')
@mock.patch('src.api.models.model.Model.load_output_configs')
def test_model_image_preprocessing_exception(mock_load_model_config,
        mock_load_input_configs, mock_load_output_configs, exceptions):
    ovms_connector = OvmsConnector("4000",
            {"model_name": "test", "model_version": 1, "input_name": "input"})
    mod = TestModel("test_model", ovms_connector)
    mock_load_model_config.return_value = []
    mock_load_input_configs.return_value = []
    mock_load_output_configs.return_value = []
    mod.preprocess_binary_image = mock.Mock(side_effect=exceptions)
    resp = falcon.Response()
    req = type('test', (), {})()
    req.headers = {}
    req.bounded_stream = io.BytesIO(b"")
    mod.on_post(req, resp)
    assert resp.status == falcon.HTTP_400

exceptions_and_statuses = [(ValueError, falcon.HTTP_400),
        (TypeError, falcon.HTTP_400), (ModelNotFoundError, falcon.HTTP_500),
        (OvmsUnavailableError, falcon.HTTP_503), (RequestProcessingError, falcon.HTTP_500)]
@pytest.mark.parametrize("exceptions", exceptions_and_statuses)
@mock.patch('src.api.models.model.Model.load_model_config')
@mock.patch('src.api.models.model.Model.load_input_configs')
@mock.patch('src.api.models.model.Model.load_output_configs')
def test_model_inference_exceptions(mock_load_model_config,
        mock_load_input_configs, mock_load_output_configs, exceptions):
    ovms_connector = OvmsConnector("4000",
        {"model_name": "test", "model_version": 1, "input_name": "input"})
    ovms_connector.send = mock.Mock(
            side_effect=exceptions[0])
    mod = TestModel("test_model", ovms_connector)
    mock_load_model_config.return_value = []
    mock_load_input_configs.return_value = []
    mock_load_output_configs.return_value = []
    mod.preprocess_binary_image = mock.Mock()
    resp = falcon.Response()
    req = type('test', (), {})()
    req.headers = {}
    req.bounded_stream = io.BytesIO(b"")
    mod.on_post(req, resp)
    assert resp.status == exceptions[1]

