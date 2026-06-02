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

import argparse

import mteb
from mteb.models.model_implementations.openai_models import OpenAIModel
from openai import OpenAI

parser = argparse.ArgumentParser(
    description="Run MTEB benchmark against OVMS embeddings endpoint")
parser.add_argument('--service_url', required=False, default='http://localhost:8000/v3/embeddings',
                    help='Specify url to embeddings endpoint.',
                    dest='service_url')
parser.add_argument('--model_name', default='Alibaba-NLP/gte-large-en-v1.5',
                    help='Model name to query. default: Alibaba-NLP/gte-large-en-v1.5',
                    dest='model_name')
parser.add_argument('--dataset', default='Banking77Classification',
                    help='Dataset to benchmark. default: Banking77Classification',
                    dest='dataset')
parser.add_argument('--results_dir', default='results', help='Results directory. default: results',
                    dest='results_dir')
parser.add_argument('--verbosity', choices=['0', '1', '2', '3'], default='3', )
parser.add_argument('--overwrite_results', action='store_true', default=True, )
parser.add_argument('--eval_splits', nargs='*', default=None,
                    help='Evaluation splits to use, e.g. --eval_splits test dev. If not set, '
                         'all splits defined in the task are used.',
                    dest='eval_splits')
parser.add_argument('--embed_dim', type=int, default=None,
                    help='Embedding dimension. Auto-detected if not provided.',
                    dest='embed_dim')
parser.add_argument('--max_tokens', type=int, default=99999,
                    help='Max input tokens for truncation. default: 99999',
                    dest='max_tokens')

args = vars(parser.parse_args())

client = OpenAI(base_url=args['service_url'], api_key="unused")

embed_dim = args['embed_dim']
if embed_dim is None:
    resp = client.embeddings.create(input=['dim probe'], model=args['model_name'])
    embed_dim = len(resp.data[0].embedding)

model = OpenAIModel(
    model_name=args['model_name'],
    max_tokens=args['max_tokens'],
    embed_dim=embed_dim,
    client=client,
)
tasks = mteb.get_task(args['dataset'], eval_splits=args["eval_splits"])
evaluation = mteb.MTEB(tasks=[tasks])
results = evaluation.run(model=model,
                         verbosity=int(args['verbosity']),
                         overwrite_results=args['overwrite_results'],
                         output_folder=args["results_dir"]
                         )
for task_results in results:
    # Banking77Classification: 0.8495 mean accuracy
    print(f"{task_results.task_name}: {task_results.get_score():.4f} mean {task_results.task.metadata.main_score}")

# For full leaderboard tests set run:
# benchmark = mteb.get_benchmark("MTEB(eng)")
# evaluation = mteb.MTEB(tasks=benchmark)
# evaluation.run(model, verbosity=3, overwrite_results=True, output_folder='results')

