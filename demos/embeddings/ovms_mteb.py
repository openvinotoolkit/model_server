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

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import numpy as np
import mteb
from mteb.model_meta import ModelMeta
logger = logging.getLogger(__name__)
import argparse

parser = argparse.ArgumentParser(description='Compare embeddings responses from HF transformers, OVSentenceTransformer and OVMS')
parser.add_argument('--service_url', required=False, default='http://localhost:6000/v3/embeddings',
                    help='Specify url to embeddings endpoint. default:http://localhost:8000/v3/embeddings', dest='service_url')
parser.add_argument('--model_name', default='Alibaba-NLP/gte-large-en-v1.5', help='Model name to query. default: Alibaba-NLP/gte-large-en-v1.5',
                    dest='model_name')
args = vars(parser.parse_args())


class OVMSModel:
    def __init__(self, model_name: str, base_url:str, embed_dim: int | None = None, **kwargs) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url,api_key="unused")
        self._model_name = model_name
        self._embed_dim = embed_dim

    def encode(
        self, sentences: list[str], **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        max_batch_size = 32
        sublists = [
            sentences[i : i + max_batch_size]
            for i in range(0, len(sentences), max_batch_size)
        ]
        all_embeddings = []
        for sublist in sublists:
            response = self._client.embeddings.create(
                input=sublist,
                model=self._model_name,
                encoding_format="float",
                dimensions=self._embed_dim or NotGiven(),
            )
            all_embeddings.extend(self._to_numpy(response))

        return np.array(all_embeddings)
    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries, **kwargs)


    def _to_numpy(self, embedding_response) -> np.ndarray:
        return np.array([e.embedding for e in embedding_response.data])

model = OVMSModel(args['model_name'], args['service_url'] ,1)
tasks = mteb.get_task("Banking77Classification")
evaluation = mteb.MTEB(tasks=[tasks])
evaluation.run(model,verbosity=3,overwrite_results=True,output_folder='results')
# For full leaderboard tests set run:
# benchmark = mteb.get_benchmark("MTEB(eng)")
# evaluation = mteb.MTEB(tasks=benchmark)
# evaluation.run(model,verbosity=3,overwrite_results=True,output_folder='results')

