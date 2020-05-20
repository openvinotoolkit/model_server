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

from src.logger import get_logger
from src.constants import TYPE_CLASS_MAPPING
from src.api.ovms_connector import OvmsConnector
from src.api.models.model_config import ModelConfigurationSchema, \
    ModelInputConfiguration, ModelInputConfigurationSchema, ValidationError, \
    ModelOutputConfiguration, ModelOutputConfigurationSchema

logger = get_logger(__name__)


class ModelBuilder:

    @classmethod
    def build_model(cls, config_file_path, ovms_port):
        model_config = cls.load_model_config(config_file_path)
        model_type = model_config.pop('model_type')
        model_config['ovms_connector'] = OvmsConnector(
            ovms_port, model_config.pop('ovms_mapping'))
        model = TYPE_CLASS_MAPPING[model_type](**model_config)
        return model

    @classmethod
    def load_model_config(cls, config_file_path: str) -> dict:
        """
        :raises ValueError: when loading of configuration file fails
        :raises ValidationError: when cofiguration is incomplete or invalid
        """
        try:
            with open(config_file_path, mode='r') as config_file:
                config = json.load(config_file)
        except FileNotFoundError as e:
            logger.exception('Model\'s configuration file {} was not found.'.format(config_file_path))
            raise FileNotFoundError from e
        except Exception as e:
            logger.exception(
                'Failed to load Model\'s configuration file {}.'.format(config_file_path))
            raise ValueError from e

        model_config_schema = ModelConfigurationSchema()
        errors_dict = model_config_schema.validate(config)
        if errors_dict:
            msg = 'Model configuration is invalid: {}'.format(errors_dict)
            logger.exception(msg)
            raise ValidationError(msg)
        model_config = {}
        model_config['endpoint'] = config['endpoint']
        model_config['model_type'] = config['model_type']
        model_config['ovms_mapping'] = config['ovms_mapping']
        model_config['input_configs'] = cls._load_input_configs(config)
        model_config['output_configs'] = cls._load_output_configs(config)

        return model_config

    @classmethod
    def _load_input_configs(cls, model_config: dict) -> Dict[str, ModelInputConfiguration]:
        """
        :raises ValueError: when loading of configuration file fails
        :raises marshmallow.ValidationError: if input configuration has invalid schema
        :returns: a dictionary where key is the input name and value
                 is ModelInputConfiguration for given input
        """
        model_input_configs = {}
        input_config_schema = ModelInputConfigurationSchema()

        for input_config_dict in model_config.get('inputs', []):
            try:
                input_config = input_config_schema.load(input_config_dict)
            except ValidationError:
                logger.exception(
                    'Model input configuration is invalid: {}'.format(input_config_dict))
                raise

            model_input_configs[input_config.input_name] = input_config
            logger.info('Loaded model input configuration: {}'.format(
                input_config_dict))

        return model_input_configs

    @staticmethod
    def _load_output_configs(model_config: dict) -> Dict[str, ModelOutputConfiguration]:
        """
        :raises ValueError: when loading of configuration file fails
        :raises marshmallow.ValidationError: if output configuration has invalid schema
        :returns: a dictionary where key is the output name and value
                 is ModelOutputConfiguration for given output
        """
        model_output_configs = {}
        output_config_schema = ModelOutputConfigurationSchema()

        for output_config_dict in model_config.get('outputs', []):
            try:
                output_config = output_config_schema.load(output_config_dict)
            except ValidationError:
                logger.exception(
                    'Model output configuration is invalid: {}'.format(output_config_dict))
                raise

            model_output_configs[output_config.output_name] = output_config
            logger.info('Loaded model output configuration: {}'.format(
                output_config_dict))

        return model_output_configs
