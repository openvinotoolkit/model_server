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

class ServingClient(ABC):

    @abstractmethod
    def predict(self, request):
        '''
        Send PredictRequest to the server and return response.

        Args:
            request: PredictRequest object.

        Returns:
            PredictResponse object

        Raises:
            TypeError:  if provided argument is of wrong type.
            Many more for different serving reponses...
        '''

        pass

    @abstractmethod
    def get_model_metadata(self, request):
        '''
        Send ModelMetadataRequest to the server and return response.

        Args:
            request: ModelMetadataRequest object.

        Returns:
            ModelMetadataResponse object

        Raises:
            TypeError:  if provided argument is of wrong type.
            Many more for different serving reponses...
        '''

        pass

    @abstractmethod
    def get_model_status(self, request):
        '''
        Send ModelStatusRequest to the server and return response.

        Args:
            request: ModelStatusRequest object.

        Returns:
            ModelStatusResponse object

        Raises:
            TypeError:  if provided argument is of wrong type.
            Many more for different serving reponses...
        '''

        pass

    @classmethod
    @abstractmethod
    def _build(cls, config):
        raise NotImplementedError
