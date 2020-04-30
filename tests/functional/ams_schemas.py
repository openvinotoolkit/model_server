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
    l = fields.Float(required=True)
    t = fields.Float(required=True)
    w = fields.Float(required=True)
    h = fields.Float(required=True)


class ClassificationSchema(Schema):
    tag = fields.Nested(TagSchema)
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
    # TODO: make sure that type is corresponding to proper nested schema
    inference_type = fields.String(required=True,
                                   data_key='type',
                                   validate=validate.OneOf({'classification',
                                                            'motion',
                                                            'entity',
                                                            'text',
                                                            'other',
    }))
    subtype = fields.String(required=True)
    classification = fields.Nested(ClassificationSchema, required=False) 
    motion = fields.Nested(MotionSchema, required=False) 
    entity = fields.Nested(EntitySchema, required=False) 
    text = fields.Nested(TextSchema, required=False) 
    extensions = fields.Dict(required=False)

    @validates_schema
    def validate_type(self, data, **kwargs):
        if not data.get(data.get('type')):
            raise ValidationError('Inference response content {} does '
                                  'not match declared response type.'.format(data))
