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
from optimum.intel import OVModelForFeatureExtraction, OVSentenceTransformer
import torch
import datetime
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Compare embeddings responses from HF transformers, OVSentenceTransformer and OVMS')
parser.add_argument('--service_url', required=False, default='http://localhost:8000/v3/embeddings',
                    help='Specify url to embeddings endpoint. default:http://localhost:8000/v3/embeddings', dest='service_url')
parser.add_argument('--model_name', default='Alibaba-NLP/gte-large-en-v1.5', help='Model name to query. default: Alibaba-NLP/gte-large-en-v1.5',
                    dest='model_name')
parser.add_argument('--input', default=[], help='List of strings to query. default: []',
                    dest='input', action='append')
parser.add_argument('--pooling', default="CLS", choices=["CLS", "LAST"], help='Embeddings pooling mode', dest='pooling')

args = vars(parser.parse_args())

model_id = args['model_name']
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model_pt = AutoModel.from_pretrained(model_id, trust_remote_code=True)
#model_ov = OVSentenceTransformer.from_pretrained(model_id, trust_remote_code=True)

text = args['input']
print("input", text)

def run_model():
    with torch.no_grad():
        start_time = datetime.datetime.now()
        input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        model_output = model_pt(**input)
        if args['pooling'] == "LAST_TOKEN":
            sequence_lengths = input['attention_mask'].sum(dim=1) - 1
            batch_size = model_output.last_hidden_state.shape[0]
            embeddings = model_output.last_hidden_state[torch.arange(batch_size, device=model_output.last_hidden_state.device), sequence_lengths]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        else:
            embeddings = model_output.last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        print("HF Duration:", duration, "ms", type(model_pt).__name__)
        return np.array(embeddings)

def run_OV():
    with torch.no_grad():
        start_time = datetime.datetime.now()
        embeddings = model_ov.encode(text)
        embeddings = embeddings / np.sqrt(np.sum(embeddings**2))
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        print("OV Duration:", duration, "ms", type(model_ov).__name__)
        return embeddings

def run_ovms():
    from openai import OpenAI
    client = OpenAI(base_url=args['service_url'],api_key="unused"    )
    start_time = datetime.datetime.now()
    responses = client.embeddings.create(input=text, model=model_id)
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    print("OVMS Duration:", duration, "ms",)
    return responses.data

HF_embeddings = run_model()
#OV_embeddings = run_OV()
OVMS_embeddings = run_ovms()

i=0
for res in OVMS_embeddings:
    print("Batch number:", i)
    ovmsresult = np.array(res.embedding)
    with np.printoptions(precision=4, suppress=True):
        print("OVMS embeddings: shape:",ovmsresult.shape, "emb[:20]:\n", ovmsresult[:20])
        #print("OVSentenceTransformer: shape:",OV_embeddings[i].shape, "emb[:20]:\n", OV_embeddings[i][:20])
        print("HF AutoModel: shape:",HF_embeddings[i].shape, "emb[:20]:\n", HF_embeddings[i][:20])
    print("Difference score with HF AutoModel:", np.linalg.norm(ovmsresult - HF_embeddings[i]))
    assert np.allclose(ovmsresult, HF_embeddings[i], atol=1e-2)
    assert (np.linalg.norm(ovmsresult - HF_embeddings[i]) < 0.06)
    i+=1


