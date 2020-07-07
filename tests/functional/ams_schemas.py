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

from marshmallow import Schema, fields, validate, \
     validates_schema, ValidationError


class TagSchema(Schema):
    value = fields.String(required=True)
    confidence = fields.Float(required=True)


class AttributeSchema(Schema):
    name = fields.String(required=True)
    value = fields.String(required=True)
    confidence = fields.Float(required=True)


class BoxSchema(Schema):
    l = fields.Float(required=True)  # noqa: E741
    t = fields.Float(required=True)
    w = fields.Float(required=True)
    h = fields.Float(required=True)


class ClassificationSchema(Schema):
    attributes = fields.List(fields.Nested(AttributeSchema))


class MotionSchema(Schema):
    box = fields.Nested(BoxSchema)


class EntitySchema(Schema):
    tag = fields.Nested(TagSchema)
    attributes = fields.List(fields.Nested(AttributeSchema))
    box = fields.Nested(BoxSchema)


class TextSchema(Schema):
    value = fields.String(required=True)
    language = fields.String(required=True)
    startTimestamp = fields.Float(required=True)
    endTimestamp = fields.Float(required=True)


class OtherSchema(Schema):
    pass


class InferenceResponseSchema(Schema):
    inference_type = fields.String(required=True,
                                   data_key='type',
                                   validate=validate.OneOf({'classification',
                                                            'motion',
                                                            'entity',
                                                            'text',
                                                            'other',
                                                            }))
    subtype = fields.String(required=True)
    classifications = fields.List(fields.Nested(ClassificationSchema, required=False))
    motions = fields.List(fields.Nested(MotionSchema, required=False))
    entities = fields.List(fields.Nested(EntitySchema, required=False))
    texts = fields.List(fields.Nested(TextSchema, required=False))
    extensions = fields.Dict(required=False)

    @validates_schema
    def validate_type(self, data, **kwargs):
        if not data.get(data.get('type')):
            raise ValidationError('Inference response content {} does not match '
                                  'declared response type.'.format(data))
