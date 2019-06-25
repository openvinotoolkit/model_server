from ie_serving.server.constants import ROW_FORMAT, ROW_SIMPLIFIED

responses = {
    'row': {
        'full': (lambda inference_output:
                 {'predictions': _column_to_row(inference_output)}),
        'simplified': (lambda inference_output:
                       {'predictions': list(inference_output.values())[0]}),
    },
    'column': {
        'full': (lambda inference_output:
                 {'outputs': inference_output}),
        'simplified': (lambda inference_output:
                       {'outputs': list(inference_output.values())[0]})
    }
}


def _row_to_column(list_of_dicts):
    output_dict = dict()
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            if key not in output_dict.keys():
                output_dict[key] = []
            output_dict[key].append(value)
    return output_dict


def _column_to_row(dict_of_lists):
    output_list = []
    for values in zip(*dict_of_lists.values()):
        dictionary = dict()
        for (key, value) in zip(dict_of_lists.keys(), values):
            dictionary[key] = value
        output_list.append(dictionary)
    return output_list


def preprocess_json_request(request_body, input_format, model_input_keys):
    if input_format == ROW_FORMAT:
        inputs = _row_to_column(request_body['instances'])
    elif input_format == ROW_SIMPLIFIED:
        inputs = request_body['instances']
    else:
        inputs = request_body['inputs']

    if type(inputs) is list:
        inputs = {model_input_keys[0]: inputs}
    return inputs


def prepare_json_response(output_representation, inference_output):
    if len(inference_output.keys()) > 1:
        response = responses[output_representation]['full'](inference_output)
    else:
        response = responses[output_representation]['simplified'](
            inference_output)
    return response
