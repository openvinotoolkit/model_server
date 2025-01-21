#
# Copyright (c) 2024 Intel Corporation
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

from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from optimum.intel import OVModelForFeatureExtraction, OVSentenceTransformer
import torch
import datetime
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Compare rerank responses from HF transformers OVMS')
parser.add_argument('--base_url', required=False, default='http://localhost:8000/v3/',
                    help='Specify url to embeddings endpoint. default:http://localhost:8000/v3', dest='base_url')
parser.add_argument('--model_name', default='BAAI/bge-reranker-large', help='Model name to query. default: Alibaba-NLP/gte-large-en-v1.5',
                    dest='model_name')
parser.add_argument('--query', default='', help='Query string to rerank.',
                    dest='query')
parser.add_argument('--document', default=[], help='List of strings to query. default: []',
                    dest='input', action='append')
args = vars(parser.parse_args())

model_id = args['model_name']
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model_pt = AutoModelForSequenceClassification.from_pretrained(model_id,trust_remote_code=True)

query = args['query']
documents = args['input']
print("query", query)
print("documents", documents)


def run_model():
    pairs = [[query, doc] for doc in documents]
    with torch.no_grad():
        start_time = datetime.datetime.now()
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model_pt(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = list(1 / (1 + np.exp(-scores)))
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        print("HF Duration:", duration, "ms")
        return np.array(scores)

def run_ovms():
    import cohere
    client = cohere.Client(base_url=args["base_url"], api_key="not_used")
    start_time = datetime.datetime.now()
    responses = client.rerank(query=query,documents=documents, model=model_id)
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    print("OVMS Duration:", duration, "ms")
    if hasattr(responses, "results"):
        responses = getattr(responses, "results")
    result_dicts = {}
    results_list = []
    for res in responses:
        result_dicts[res.index]= res.relevance_score
    for i in range(len(result_dicts)):
        results_list.append(result_dicts[i])
    return np.array(results_list)   

HF_reranking = run_model()
#OV_embeddings = run_OV()
OVMS_reranking = run_ovms()

print("HF reranking:", HF_reranking)
print("OVMS reranking:", OVMS_reranking)



