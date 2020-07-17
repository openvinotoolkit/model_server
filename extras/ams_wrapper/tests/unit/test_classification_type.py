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

from src.api.types import SingleClassification, Classification, Attribute


def test_single_classification():
    expected_dict = {
        "type": "classification",
        "subtype": "type",
        "classification": {
            "tag": "car"
        }
    }
    attribute = Attribute('car', 0.97)
    attribs = []
    attribs.append(attribute)
    test_classification = SingleClassification(subtype_name='type', attributes=attribs)
    assert expected_dict == test_classification.as_dict()
    print(test_classification.as_dict())


def test_entity():
    expected_dict = {
        "inferences": [
            {
                "type": "classification",
                "subtype": "animal",
                "classification": {
                    'confidence': 0.85,
                    "tag": "dog"
                }
            },
            {
                "type": "classification",
                "subtype": "animal",
                "classification": {
                    'confidence': 0.11,
                    "tag": "fox"
                }
            }
        ]
    }

    classifications = [
        SingleClassification(subtype_name='animal', attributes=[Attribute("dog", 0.85)]),
        SingleClassification(subtype_name='animal', attributes=[Attribute("fox", 0.11)]),
    ]
    classification = Classification(classifications)
    assert expected_dict == classification.as_dict()
