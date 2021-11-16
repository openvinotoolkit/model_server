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

from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow_serving.apis import get_model_status_pb2, get_model_metadata_pb2, predict_pb2

from ovmsclient.tfs_compat.base.requests import (PredictRequest, ModelMetadataRequest,
                                                 ModelStatusRequest, _check_model_spec)
from ovmsclient.tfs_compat.grpc.tensors import make_tensor_proto


class GrpcPredictRequest(PredictRequest):
    def __init__(self, inputs, model_name, model_version, raw_request):
        super().__init__(inputs, model_name, model_version)
        self.raw_request = raw_request


class GrpcModelMetadataRequest(ModelMetadataRequest):
    def __init__(self, model_name, model_version, raw_request):
        super().__init__(model_name, model_version)
        self.raw_request = raw_request


class GrpcModelStatusRequest(ModelStatusRequest):

    def __init__(self, model_name, model_version, raw_request):
        super().__init__(model_name, model_version)
        self.raw_request = raw_request


def make_predict_request(inputs, model_name, model_version=0):
    '''
    Create GrpcPredictRequest object.

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
                          | numpy array,
                          | TensorProto
            ============  ==================

            If provided **input_data** is not TensorProto,
            the make_tensor_proto function with default parameters will be called internally.

        model_name: Name of the model that will receive the request.

        model_version (optional): Version of the model that will receive the request.
            By default this value is set to 0, meaning the request will be sent to the default version of the model.

    Returns:
        GrpcPredictRequest object filled with **inputs** and target model spec.

    Raises:
        TypeError:  if unsupported types are provided.
        ValueError: if arguments have inappropriate values.

    Examples:
        Request to the default version of the model called "model" that has 2 inputs:

        >>> predict_request = make_predict_request(
        ...     inputs={
        ...         "binary_input": bytes([1, 2, 3, 4, 5, 6]),
        ...         "numeric_input": np.array([[1, 2, 3], [4, 5, 6]], np.int32)
        ...     },
        ...     model_name="model")
        >>> print(predict_request)

        Request to the second version of the model called "model" that has 1 input.
        Providing data as TensorProto to make sure desired data type is set for the input:

        >>> data = make_tensor_proto([1, 2, 3], dtype=DataTypes.float32)
        >>> predict_request = make_predict_request(
        ...     inputs={
        ...         "input": data
        ...     },
        ...     model_name="model",
        ...     model_version=2)
        >>> print(predict_request)
    '''  # noqa: E501

    _check_model_spec(model_name, model_version)

    if not isinstance(inputs, dict):
        raise TypeError(f'inputs type should be dict, but is {type(inputs).__name__}')

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.version.value = model_version

    for input_name, input_data in inputs.items():
        if not isinstance(input_name, str):
            raise TypeError(f'inputs keys type should be str, but found '
                            f'{type(input_name).__name__}')
        if isinstance(input_data, TensorProto):
            request.inputs[input_name].CopyFrom(input_data)
        else:
            request.inputs[input_name].CopyFrom(make_tensor_proto(input_data))

    return GrpcPredictRequest(inputs, model_name, model_version, request)


def make_metadata_request(model_name, model_version=0):
    '''
    Create GrpcModelMetadataRequest object.

    Args:

        model_name: Name of the model that will receive the request.

        model_version (optional): Version of the model that will receive the request.
            By default this value is set to 0, meaning the request will be sent to the default version of the model.

    Returns:
        GrpcModelMetadataRequest object with target model spec.

    Raises:
        TypeError:  if unsupported types are provided.
        ValueError: if arguments have inappropriate values.

    Examples:
        Request to the second version of the model called "model":

        >>> metadata_request = make_metadata_request(model_name="model", model_version=2)
        >>> print(metadata_request)
    '''  # noqa: E501

    _check_model_spec(model_name, model_version)

    metadata_field = "signature_def"
    request = get_model_metadata_pb2.GetModelMetadataRequest()
    request.model_spec.name = model_name
    request.model_spec.version.value = model_version
    request.metadata_field.append(metadata_field)
    return GrpcModelMetadataRequest(model_name, model_version, request)


def make_status_request(model_name, model_version=0):
    '''
    Create GrpcModelStatusRequest object.

    Args:

        model_name: Name of the model that will receive the request.

        model_version (optional): Version of the model that will receive the request. Must be type int.
            By default this value is set to 0, meaning the request will be sent to the default version of the model.

    Returns:
        GrpcModelStatusRequest object with target model spec.

    Raises:
        TypeError:  if unsupported types are provided.
        ValueError: if arguments have inappropriate values.

    Examples:
        Request to the second version of the model called "model":

        >>> status_request = make_status_request(model_name="model", model_version=2)
        >>> print(status_request)

    '''  # noqa: E501

    _check_model_spec(model_name, model_version)

    request = get_model_status_pb2.GetModelStatusRequest()
    request.model_spec.name = model_name
    request.model_spec.version.value = model_version
    return GrpcModelStatusRequest(model_name, model_version, request)
