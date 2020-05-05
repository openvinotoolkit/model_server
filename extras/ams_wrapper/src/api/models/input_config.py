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
    
    def as_dict(self) -> dict:
        return vars(self)


class ModelInputConfigurationSchema(Schema):
    input_name = fields.String(required=True)
    channels = fields.Integer(required=False)
    target_height = fields.Integer(required=False)
    target_width = fields.Integer(required=False)
    color_format = fields.String(required=False, validate=validate.OneOf({'BGR', 'RGB'}))
    scale = fields.Float(required=False,
                         validate=validate.Range(min=0, min_inclusive=False))
    standardization = fields.Bool(required=False)
    input_format = fields.String(required=False, validate=validate.OneOf({'NCHW', 'NHWC'}))

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