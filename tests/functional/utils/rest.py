#
# Copyright (c) 2019 Intel Corporation
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

import numpy as np
import requests
from google.protobuf.json_format import Parse
from tensorflow_serving.apis import get_model_metadata_pb2, \
    get_model_status_pb2


def prepare_body_format(img, request_format, input_name):
    signature = "serving_default"
    if request_format == "row_name":
        instances = []
        for i in range(0, img.shape[0], 1):
            instances.append({input_name: img[i].tolist()})
        data_obj = {"signature_name": signature, "instances": instances}
    elif request_format == "row_noname":
        data_obj = {"signature_name": signature, 'instances': img.tolist()}
    elif request_format == "column_name":
        data_obj = {"signature_name": signature,
                    'inputs': {input_name: img.tolist()}}
    elif request_format == "column_noname":
        data_obj = {"signature_name": signature, 'inputs': img.tolist()}
    data_json = json.dumps(data_obj)
    return data_json


def process_json_output(result_dict, output_tensors):
    output = {}
    if "outputs" in result_dict:
        keyname = "outputs"
        if type(result_dict[keyname]) is dict:
            for output_tensor in output_tensors:
                output[output_tensor] = np.asarray(
                    result_dict[keyname][output_tensor])
        else:
            output[output_tensors[0]] = np.asarray(result_dict[keyname])
    elif "predictions" in result_dict:
        keyname = "predictions"
        if type(result_dict[keyname][0]) is dict:
            for row in result_dict[keyname]:
                print(row.keys())
                for output_tensor in output_tensors:
                    if output_tensor not in output:
                        output[output_tensor] = []
                    output[output_tensor].append(row[output_tensor])
            for output_tensor in output_tensors:
                output[output_tensor] = np.asarray(output[output_tensor])
        else:
            output[output_tensors[0]] = np.asarray(result_dict[keyname])
    else:
        print("Missing required response in {}".format(result_dict))

    return output


def infer_rest(img, input_tensor, rest_url,
               output_tensors, request_format):
    data_json = prepare_body_format(img, request_format, input_tensor)
    result = requests.post(rest_url, data=data_json)
    output_json = json.loads(result.text)
    data = process_json_output(output_json, output_tensors)
    return data


def get_model_metadata_response_rest(rest_url):
    result = requests.get(rest_url)
    output_json = result.text
    metadata_pb = get_model_metadata_pb2.GetModelMetadataResponse()
    response = Parse(output_json, metadata_pb, ignore_unknown_fields=False)
    return response


def get_model_status_response_rest(rest_url):
    result = requests.get(rest_url)
    output_json = result.text
    status_pb = get_model_status_pb2.GetModelStatusResponse()
    response = Parse(output_json, status_pb, ignore_unknown_fields=False)
    return response
