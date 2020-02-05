#
# Copyright (c) 2019-2020 Intel Corporation
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

latest_schema = {
    'type': 'object',
    'required': ['latest'],
    'properties': {'latest': {
        'type': 'object',
        'properties': {'num_versions': {
            'type': 'integer',
        }},
    }},
}

versions_schema = {
    'type': 'object',
    'required': ['specific'],
    'properties': {'specific': {
        'type': 'object',
        'required': ['versions'],
        'properties': {
            'versions': {
                'type': 'array',
                'items': {
                    'type': 'integer',
                },
            }},
    }},
}


all_schema = {
    'type': 'object',
    'required': ['all'],
    'properties': {'all': {'type': 'object'}},
}

models_config_schema = {
    'definitions': {
        'model_config': {
            'type': 'object',
            'required': ['config'],
            'properties': {
                'config': {
                    'type': 'object',
                    'required': ['name', 'base_path'],
                    'properties': {
                        'name': {'type': 'string'},
                        'base_path': {'type': 'string'},
                        'batch_size': {'type': ['integer', 'string']},
                        'model_version_policy': {'type': 'object'},
                        'shape': {'type': ['object', 'string']},
                        'nireq': {'type': 'integer'},
                        'target_device': {'type': 'string'},
                        'plugin_config': {'type': 'object'}
                    }
                }
            }
        }
    },
    'type': 'object',
    'required': ['model_config_list'],
    'properties': {'model_config_list': {
        'type': 'array',
        'items': {"$ref": "#/definitions/model_config"},
    }},
}
