#
# Copyright (c) 2026 Intel Corporation
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

# pylint: disable=unused-argument


from dataclasses import dataclass
from typing import Union

from tests.functional.utils.inference.serving.common import LLMCommonWrapper
from tests.functional.utils.logger import get_logger

logger = get_logger(__name__)

COHERE = "COHERE"


class CohereWrapper(LLMCommonWrapper):
    API_KEY_NOT_USED = "not_used"
    RERANK = "rerank"
    PREDICT = RERANK

    @staticmethod
    def prepare_body_dict(input_objects: dict, request_format=None, **kwargs):
        model = kwargs.get("model", None)
        assert model is not None, "No model provided"
        model_name = model.name

        endpoint = kwargs.get("endpoint", CohereWrapper.RERANK)
        body_dict = {}
        if endpoint == CohereWrapper.RERANK:
            documents = [f'"{doc}"' for doc in input_objects["input0"]["documents"]]
            body_dict = {
                "model": model_name,
                "query": input_objects["input0"]["query"],
                "documents": documents
            }
        else:
            raise NotImplementedError(f"Invalid endpoint: {endpoint}")
        return LLMCommonWrapper.prepare_body_dict_from_request_params(CohereRequestParams, body_dict, **kwargs)

    def get_model_meta_grpc_request(self, model_name=None):
        raise NotImplementedError

    def get_predict_grpc_request(self):
        raise NotImplementedError

    def set_grpc_stubs(self):
        raise NotImplementedError


class RerankApi:

    @staticmethod
    def prepare_rerank_input_content(input_objects):
        for input_name in input_objects:
            return input_objects[input_name]


@dataclass
class CohereRequestParams:

    def prepare_dict(self, **kwargs):
        request_params_dict = {key: value for key, value in vars(self).items() if value is not None}
        return request_params_dict

class CohereRerankRequestParams(CohereRequestParams):
    documents: Union[str, list] = None
    top_n: int = None
    return_documents: bool = None

    def set_default_values(self):
        self.top_n = 1000
        self.return_documents = False


@dataclass
class OvmsRerankRequestParams(CohereRerankRequestParams):
    pass
