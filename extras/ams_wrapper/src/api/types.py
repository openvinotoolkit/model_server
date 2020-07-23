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
    def __init__(self, value: str, confidence: float):
        self.value = value
        self.confidence = confidence

    def as_dict(self):
        result_dict = {
            "value": self.value,
            "confidence": self.confidence
        }
        return result_dict


class Rectangle:
    def __init__(self, l: float, t: float, w: float, h: float):
        self.l = l  # noqa: E741
        self.t = t
        self.w = w
        self.h = h

    def as_dict(self):
        result_dict = {
            "l": self.l,
            "t": self.t,
            "w": self.w,
            "h": self.h
        }
        return result_dict


class ResultType(ABC):
    pass


class Motion(ResultType):
    def __init__(self, box: Rectangle):
        self.box = box
        self.type_name = "motion"

    def as_dict(self):
        return {
            "box": self.box.as_dict()
        }


class SingleClassification:
    def __init__(self, subtype_name: str, tag: Tag):
        self.tag = tag
        self.type_name = "classification"
        self.subtype_name = subtype_name

    def as_dict(self):
        result_dict = {
            "type": self.type_name,
            "subtype": self.subtype_name,
            "classification": {
                "tag": self.tag.as_dict()
            }
        }
        return result_dict


class Classification(ResultType):
    def __init__(self, classifications: List[SingleClassification]):
        self.classifications = classifications

    def as_dict(self):
        result_dict = {
            "inferences": [classification.as_dict() for classification in self.classifications]
        }
        return result_dict


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
            result_dict["attributes"] = [attribute.as_dict()
                                         for attribute in self.attributes]
        return result_dict


class Entity:
    def __init__(self, subtype_name: str, entity: SingleEntity):
        self.type_name = "entity"
        self.subtype_name = subtype_name
        self.entity = entity

    def as_dict(self):
        result_dict = {
                "type": self.type_name,
                "subtype": self.subtype_name,
                "entity": self.entity.as_dict()
        }
        return result_dict


class Detection(ResultType):
    def __init__(self, entities: List[Entity]):
        self.entities = entities

    def as_dict(self):
        result_dict = {
            "inferences": [entity.as_dict() for entity in self.entities]
        }
        return result_dict
