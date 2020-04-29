from typing import List
from abc import ABC

class Tag:
    def __init__(self, value: str, confidence: float):
        self.value = value
        self.confidence = confidence

    def as_dict(self):
        result_dict = {
            "value": self.value,
            "confidence": self.confidence
        }
        return result_dict

class Attribute:
    def __init__(self, name: str, value: str, confidence: float):
        self.name = name
        self.value = value
        self.confidence = confidence

    def as_dict(self):
        result_dict = {
            "name": self.name
            "value": self.value,
            "confidence": self.confidence
        }
        return result_dict

class Rectangle:
    def __init__ (self, l: float, t: float, r: float, b: float):
        self.l = l
        self.t = t
        self.r = r
        self.b = b

    def as_dict(self):
        result_dict = {
            "l": self.l,
            "t": self.t,
            "r": self.r,
            "b": self.b
        }
        return result_dict

class ResultType(ABC):
    pass

class Motion(ResultType):
    def __init__(self, box: Rectangle):
        self.box = box
        self.type_name = "motion"

    def as_dict(self):
        result_dict = {
            "box": self.box.as_dict()
        }

class Classification(ResultType):
    def __init__(self, tag: Tag, attributes: List[Attribute]):
        self.tag = tag
        self.attributes = attributes
        self.type_name = "classification"

    def as_dict(self):
        result_dict = {
            "tag": self.tag.as_dict(),
            "attributes": [attribute.as_dict() for attribute in self.attributes]
        }

class SingleEntity:
    def __init__(self, tag: Tag, box: Rectangle, attributes: List[Attribute] = None):
        self.tag = tag
        self.box = box
        self.attributes = attributes

    def as_dict(self):
        result_dict = {
            "tag": self.tag.as_dict(),
            "box": self.box.as_dict()
        }
        if self.attributes is not None:
            result_dict[attributes] = [attribute.as_dict() for attribute in self.attributes]
        return result_dict

class Entity(ResultType):

    def __init__(self, subtype_name: str, entities: List[SingleEntity]):
        self.type_name = "entity"
        self.subtype_name = subtype_name
        self.entities = entities

    def as_dict(self):
        result_dict = {
            "type": self.type_name,
            "subtype": self.subtype_name,
            "entities": [entity.as_dict() for entity in self.entities]
        }
        return result_dict

class Text(ResultType):
    # TODO: add support
    pass

class InferenceExtension:
    # TODO: add support
    pass

class Inference:
    def __init__(self, value: ResultType, extensions: List[InferenceExtension] = None):
        self.value = value
        self.extensions = extensions