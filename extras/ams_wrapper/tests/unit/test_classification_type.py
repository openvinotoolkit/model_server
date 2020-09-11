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

from src.api.types import SingleClassification, Classification, Tag


def test_single_classification():
    expected_dict = {
                        'type': 'classification',
                        'subtype': 'type',
                        'classification': {
                            'tag': {
                                'value': 'car',
                                'confidence': 0.97
                            }
                        }
                    }
    tag = Tag('car', 0.97)
#    attribs = []
#    attribs.append(attribute)
    test_classification = SingleClassification(subtype_name='type', tag=tag)
    print(test_classification.as_dict())
    assert expected_dict == test_classification.as_dict()


def test_entity():
    expected_dict = {
                        "inferences": [
                            {
                                "type": "classification",
                                "subtype": "animal",
                                "classification": {
                                    "tag": {
                                        "value": "dog",
                                        "confidence": 0.85
                                    }
                                }
                            },
                            {
                                "type": "classification",
                                "subtype": "animal",
                                "classification": {
                                    "tag": {
                                        "value": "fox",
                                        "confidence": 0.11
                                    }
                                }
                            }
                        ]
                    }

    classifications = [
        SingleClassification(subtype_name='animal', tag=Tag("dog", 0.85)),
        SingleClassification(subtype_name='animal', tag=Tag("fox", 0.11)),
    ]
    classification = Classification(classifications)
    assert expected_dict == classification.as_dict()
