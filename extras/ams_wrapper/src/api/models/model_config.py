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

from marshmallow import Schema, fields, validates_schema, ValidationError, post_load, validate


class ModelOvmsMappingSchema(Schema):
    model_name = fields.String(required=True)
    model_version = fields.Integer(required=True)


class ModelInputConfiguration:
    def __init__(self, input_name: str, channels: int = None,
                 target_height: int = None, target_width: int = None,
                 color_format: str = 'BGR', scale: float = None,
                 standardization: bool = False, input_format: str = 'NCHW'):
        self.input_name = input_name
        self.channels = channels
        self.target_height = target_height
        self.target_width = target_width
        self.reverse_input_channels = True if color_format == 'BGR' else False
        self.scale = scale
        self.standardization = standardization
        self.channels_first = False if input_format == 'NHWC' else True

    def as_preprocessing_options(self) -> dict:
        return {
            'channels': self.channels,
            'target_size': (self.target_height, self.target_width)
            if self.target_height and self.target_width else None,
            'channels_first': self.channels_first,
            'scale': self.scale,
            'standardization': self.standardization,
            'reverse_input_channels': self.reverse_input_channels
        }


class ModelInputConfigurationSchema(Schema):
    input_name = fields.String(required=True)
    channels = fields.Integer(required=False)
    target_height = fields.Integer(required=False)
    target_width = fields.Integer(required=False)
    color_format = fields.String(
        required=False, validate=validate.OneOf({'BGR', 'RGB'}))
    scale = fields.Float(required=False,
                         validate=validate.Range(min=0, min_inclusive=False))
    standardization = fields.Bool(required=False)
    input_format = fields.String(
        required=False, validate=validate.OneOf({'NCHW', 'NHWC'}))

    @post_load
    def make_model_input_configuration(self, data, **kwargs):
        return ModelInputConfiguration(**data)

    @validates_schema
    def validate_type(self, data, **kwargs):
        if data.get('target_width') and not data.get('target_height'):
            raise ValidationError('target_height must defined if target_width was set. '
                                  'Invalid config: {}'.format(data))
        if data.get('target_height') and not data.get('target_width'):
            raise ValidationError('target_width must defined if target_height was set. '
                                  'Invalid config: {}'.format(data))


class ModelOutputConfiguration:
    def __init__(self, output_name: str, value_index_mapping: dict = None,
                 classes: dict = None, confidence_threshold: float = None,
                 top_k_results: int = None, is_softmax=None, value_multiplier=None):
        self.output_name = output_name
        self.value_index_mapping = value_index_mapping
        self.classes = classes
        self.confidence_threshold = confidence_threshold
        self.top_k_results = top_k_results
        self.is_softmax = is_softmax
        self.value_multiplier = value_multiplier

    def __str__(self):
        return 'ModelOutputConfiguration({})'.format(vars(self))

    def __repr__(self):
        return 'ModelOutputConfiguration({})'.format(vars(self))


class ModelOutputConfigurationSchema(Schema):
    output_name = fields.String(required=True)
    is_softmax = fields.Boolean(required=False)
    value_multiplier = fields.Float(required=False)
    value_index_mapping = fields.Dict(
        keys=fields.String(), values=fields.Integer(), required=False)
    classes = fields.Dict(keys=fields.String(),
                          values=fields.Number(), required=False)
    confidence_threshold = fields.Float(
        required=False, validate=validate.Range(min=0, max=1))
    top_k_results = fields.Integer(
        required=False, validate=validate.Range(min=0, min_inclusive=False))

    @post_load
    def make_model_output_configuration(self, data, **kwargs):
        return ModelOutputConfiguration(**data)


class ModelConfigurationSchema(Schema):
    endpoint = fields.String(required=True)
    model_type = fields.String(required=True)
    inputs = fields.List(fields.Nested(ModelInputConfigurationSchema, required=True), required=True)
    outputs = fields.List(fields.Nested(ModelOutputConfigurationSchema, required=True), required=True)
    ovms_mapping = fields.Nested(ModelOvmsMappingSchema, required=True)
