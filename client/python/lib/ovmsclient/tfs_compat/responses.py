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

from abc import ABC, abstractmethod

class PredictResponse(ABC):
    
    def __init__(self, raw_response):
        self.raw_response = raw_response

class ModelMetadataResponse(ABC):
    
    def __init__(self, raw_response):
        self.raw_response = raw_response
    
    @abstractmethod
    def to_dict(self):
        '''
        Return metadata in dictionary format:
        
        .. code-block::

            {
                ...
                <version_number>: {
                    "inputs": {
                        <input_name>: {
                            "shape": <input_shape>,
                            "dtype": <input_dtype>,
                        },
                            ...
                    },
                    "outputs":
                        <output_name>: {
                            "shape": <output_shape>,
                            "dtype": <output_dtype>,
                        },
                            ...
                    }
                },
                ...
            }

        '''

        pass

class ModelStatusResponse(ABC):
    
    def __init__(self, raw_response):
        self.raw_response = raw_response
    
    @abstractmethod
    def to_dict(self):
        '''
        Return status in dictionary format:

        .. code-block::

            {
                ...
                <version_number>: {
                    "state": <model_version_state>,
                    "error_code": <error_code>,
                    "error_message": <error_message>
                },
                ...
            }

        '''

        pass
