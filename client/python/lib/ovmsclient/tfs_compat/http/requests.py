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

from enum import Enum
from ovmsclient.tfs_compat.base.requests import PredictRequest, ModelMetadataRequest, ModelStatusRequest

class HttpPredictRequest(PredictRequest):
    pass

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
    '''

    raise NotImplementedError

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
    '''

    raise NotImplementedError

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

    '''
    
    raise NotImplementedError
