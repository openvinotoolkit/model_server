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

import pytest

from start_ams import parse_ovms_model_devices_config, modify_ovms_config_json


@pytest.mark.parametrize("config_string,expected_dict", [('model_1=MULTI:MYRIAD,HDDL,CPU',
                                                          {'model_1': 'MULTI:MYRIAD,HDDL,CPU'}),
                                                         ('model_1=CPU;model_2=GPU',
                                                          {'model_1': 'CPU', 'model_2': 'GPU'}),
                                                         ])
def test_parse_ovms_model_devices_config(config_string, expected_dict):
    assert parse_ovms_model_devices_config(config_string) == expected_dict


@pytest.mark.parametrize("config_string", ['BLA', 'model_1=;model_2=;'])
def test_parse_ovms_model_devices_config_invalid(config_string):
    with pytest.raises(ValueError):
        parse_ovms_model_devices_config(config_string)


@pytest.mark.parametrize("config_string", [''])
def test_parse_ovms_model_devices_config_empty(config_string):
    assert parse_ovms_model_devices_config(config_string) == {}


@pytest.mark.parametrize("original_config,devices_config,expected_config", [
    ({"model_config_list": [
        {'config': {
            'name': 'model_1',
            'target_device': 'CPU'
        }
        },
        {'config': {
            'name': 'model_2',
            'target_device': 'CPU'
        }
        },
    ]},
        {'model_1': 'GPU'},
        {"model_config_list": [
         {'config': {
             'name': 'model_1',
             'target_device': 'GPU'
         }
         },
         {'config': {
             'name': 'model_2',
             'target_device': 'CPU'
         }
         },
         ]})
])
def test_modify_ovms_config_json(original_config, devices_config, expected_config, tmpdir):
    config_file_path = tmpdir.join('ovms_config.json')
    config_file_path = config_file_path.strpath
    with open(config_file_path, mode='w') as config_file:
        json.dump(original_config, config_file)
    print(config_file_path)
    modify_ovms_config_json(devices_config, ovms_config_path=config_file_path)

    with open(config_file_path, mode='r') as config_file:
        assert json.load(config_file) == expected_config
