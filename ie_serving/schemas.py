

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
