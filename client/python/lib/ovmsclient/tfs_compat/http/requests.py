#
# Copyright (c) 2021 Intel Corporation
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

import base64
import json
import numpy as np
from ovmsclient.tfs_compat.base.requests import (PredictRequest, ModelMetadataRequest,
                                                 ModelStatusRequest, _check_model_spec)
from ovmsclient.tfs_compat.grpc.tensors import (NP_TO_TENSOR_MAP, DataType,
                                                _get_dense_dimensions, _is_bytes_shape_valid)


class HttpPredictRequest(PredictRequest):
    def __init__(self, inputs, model_name, model_version, parsed_inputs):
        super().__init__(inputs, model_name, model_version)
        self.parsed_inputs = parsed_inputs


class HttpModelMetadataRequest(ModelMetadataRequest):
    pass


class HttpModelStatusRequest(ModelStatusRequest):
    pass


def make_predict_request(inputs, model_name, model_version=0):
    '''
    Create HttpPredictRequest object.

    Args:

        inputs: Python dictionary in format:

            .. code-block::

                {
                    ...
                    <input_name>: <input_data>,
                    ...
                }

            Following types are accepted:

            ============  ==================
            input_name    | string
            input_data    | python scalar,
                          | python list,
                          | numpy scalar,
                          | numpy array
            ============  ==================


        model_name: Name of the model that will receive the request.

        model_version (optional): Version of the model that will receive the request.
            By default this value is set to 0, meaning the request will be sent to the default version of the model.

    Returns:
        HttpPredictRequest object filled with **inputs** and target model spec.

    Raises:
        TypeError:  if unsupported types are provided.
        ValueError: if arguments have inappropriate values.

    Examples:
        Request to the default version of the model called "model" that has 2 inputs:

        >>> predict_request = make_predict_request(
        ...     inputs={
        ...         "binary_input": bytes([1, 2, 3, 4, 5, 6]),
        ...         "numeric_input: np.array([[1, 2, 3], [4, 5, 6]], np.int32)
        ...     },
        ...     model_name="model")
        >>> print(predict_request)
    '''  # noqa: E501

    _check_model_spec(model_name, model_version)

    if not isinstance(inputs, dict):
        raise TypeError(f'inputs type should be dict, but is {type(inputs).__name__}')

    parsed_inputs = {}
    for input_name, input_data in inputs.items():
        if not isinstance(input_name, str):
            raise TypeError(f'inputs keys type should be str, but found '
                            f'{type(input_name).__name__}')
        try:
            parsed_inputs[input_name] = _parse_input_data(input_data)
        except Exception as e:
            print("Error occurred when parsing input: {}. Exited with: {}".format(input_name, e))
            raise e

    parsed_inputs = json.dumps({"inputs": parsed_inputs})
    return HttpPredictRequest(inputs, model_name, model_version, parsed_inputs)


def make_metadata_request(model_name, model_version=0):
    '''
    Create HttpModelMetadataRequest object.

    Args:

        model_name: Name of the model that will receive the request.

        model_version (optional): Version of the model that will receive the request.
            By default this value is set to 0, meaning the request will be sent to the default version of the model.

    Returns:
        HttpModelMetadataRequest object with target model spec.

    Raises:
        TypeError:  if unsupported types are provided.
        ValueError: if arguments have inappropriate values.

    Examples:
        Request to the second version of the model called "model":

        >>> metadata_request = make_metadata_request(model_name="model", model_version=2)
        >>> print(metadata_request)
    '''  # noqa: E501

    _check_model_spec(model_name, model_version)
    return HttpModelMetadataRequest(model_name, model_version)


def make_status_request(model_name, model_version=0):
    '''
    Create HttpModelStatusRequest object.

    Args:

        model_name: Name of the model that will receive the request.

        model_version (optional): Version of the model that will receive the request.
            By default this value is set to 0, meaning the request will be sent to the default version of the model.

    Returns:
        HttpModelStatusRequest object with target model spec.

    Raises:
        TypeError:  if unsupported types are provided.
        ValueError: if arguments have inappropriate values.

    Examples:
        Request to the second version of the model called "model":

        >>> status_request = make_status_request(model_name="model", model_version=2)
        >>> print(status_request)

    '''  # noqa: E501
    _check_model_spec(model_name, model_version)
    return HttpModelStatusRequest(model_name, model_version)


def _parse_input_data(values):

    # create numpy ndarray from values and find its dtype if not provided
    if isinstance(values, np.ndarray):
        tensor_values = values
    elif isinstance(values, list):
        dense_dimensions = _get_dense_dimensions(values)
        tensor_values = np.array(values)
        if(list(tensor_values.shape) != dense_dimensions):
            raise ValueError(f'argument must be a dense tensor: {values} - got shape '
                             f'{list(tensor_values.shape)}, but wanted {dense_dimensions}')
    elif np.isscalar(values):
        tensor_values = np.array([values])
    else:
        raise TypeError("values type should be (list, np.ndarray, scalar), but is "
                        f"{type(values).__name__}")

    tensor_type = NP_TO_TENSOR_MAP.get(tensor_values.dtype.type)
    if tensor_type is not None:
        dtype = tensor_type.TensorDtype
    else:
        raise TypeError("provided values type is not valid")

    if dtype == DataType.DT_STRING and _is_bytes_shape_valid(tensor_values.shape, tensor_values):
        raise ValueError("bytes values with dtype DT_STRING must be in shape [N]")

    if dtype == DataType.DT_STRING:
        b64_values = []
        for value in tensor_values:
            b64_value = base64.b64encode(value).decode('utf-8')
            b64_values.append({"b64": b64_value})
        return b64_values
    else:
        return tensor_values.tolist()
