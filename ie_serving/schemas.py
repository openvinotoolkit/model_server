

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
