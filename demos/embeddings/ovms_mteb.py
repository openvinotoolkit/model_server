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
import random
from typing import Any

import numpy as np
import mteb
from datasets import DatasetDict
logger = logging.getLogger(__name__)
import argparse

parser = argparse.ArgumentParser(description='Compare embeddings responses from HF transformers, OVSentenceTransformer and OVMS')
parser.add_argument('--service_url', required=False, default='http://localhost:6000/v3/embeddings',
                    help='Specify url to embeddings endpoint. default:http://localhost:8000/v3/embeddings', dest='service_url')
parser.add_argument('--model_name', default='Alibaba-NLP/gte-large-en-v1.5', help='Model name to query. default: Alibaba-NLP/gte-large-en-v1.5',
                    dest='model_name')
parser.add_argument('--dataset', default='Banking77Classification', help='Dataset to benchmark. default: Banking77Classification',
                    dest='dataset')
parser.add_argument('--eval_splits', nargs='*', default=None,
                    help='Evaluation splits to use, e.g. --eval_splits test dev. If not set, all splits defined in the task are used.',
                    dest='eval_splits')
parser.add_argument('--hf_subsets', nargs='*', default=None,
                    help='HuggingFace dataset subsets to evaluate on, e.g. --hf_subsets en fr. '
                         'Useful for multilingual datasets to test only selected language subsets.',
                    dest='hf_subsets')
parser.add_argument('--max_samples', type=int, default=None,
                    help='Maximum number of samples to use per split. '
                         'When set, each evaluation split is truncated to at most this many samples, '
                         'allowing quick smoke-test runs on large datasets.',
                    dest='max_samples')
args = vars(parser.parse_args())


def truncate_task_datasets(task, max_samples: int, seed: int = 42) -> None:
    """Truncate every split of every subset in a loaded task to at most *max_samples* rows.

    Works on the task.dataset object in-place after task.load_data() has been called.
    Handles both multilingual layout (subset -> DatasetDict) and flat layout (DatasetDict).
    """
    rng = random.Random(seed)

    def _truncate_split(dataset, n):
        if len(dataset) <= n:
            return dataset
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        return dataset.select(sorted(indices[:n]))

    if isinstance(task.dataset, dict):
        for key in task.dataset:
            value = task.dataset[key]
            if isinstance(value, DatasetDict):
                # Multilingual: subset_name -> DatasetDict(split -> Dataset)
                for split in value:
                    value[split] = _truncate_split(value[split], max_samples)
            else:
                # Flat: split -> Dataset
                task.dataset[key] = _truncate_split(value, max_samples)


class OVMSModel:
    def __init__(self, model_name: str, base_url:str, embed_dim: int | None = None, **kwargs) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url,api_key="unused")
        self._model_name = model_name
        self._embed_dim = embed_dim

    def encode(
        self, sentences: list[str], **kwargs: Any
    ) -> np.ndarray:
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
task = mteb.get_task(args['dataset'],
                     eval_splits=args['eval_splits'],
                     hf_subsets=args['hf_subsets'])

# If --max_samples is set, load the data early and truncate before evaluation
if args['max_samples'] is not None:
    task.load_data()
    truncate_task_datasets(task, args['max_samples'])
    logger.info("Truncated dataset splits to at most %d samples", args['max_samples'])

evaluation = mteb.MTEB(tasks=[task])
evaluation.run(model,verbosity=3,overwrite_results=True,output_folder='results')
# For full leaderboard tests set run:
# benchmark = mteb.get_benchmark("MTEB(eng)")
# evaluation = mteb.MTEB(tasks=benchmark)
# evaluation.run(model,verbosity=3,overwrite_results=True,output_folder='results')

