from transformers import AutoTokenizer, AutoModel
from optimum.intel import OVModelForFeatureExtraction, OVSentenceTransformer
import torch
import datetime
import numpy as np

model_id = "Alibaba-NLP/gte-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model_pt = AutoModel.from_pretrained(model_id, trust_remote_code=True)
model_ov = OVSentenceTransformer.from_pretrained('models/gte-large-en-v1.5-embeddings/1', trust_remote_code=True, normalize=True)

text = "hello world"

def run_model(model):
    with torch.no_grad():
        start_time = datetime.datetime.now()
        input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        model_output = model(**input)
        print(model_output)
        embeddings = model_output.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        print(embeddings)
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        print("HF Duration:", duration, "ms", type(model).__name__)
        return embeddings

def run_OV():
    with torch.no_grad():
        start_time = datetime.datetime.now()
        embeddings = model_ov.encode([text])
        embeddings = embeddings / np.sqrt(np.sum(embeddings**2))
        print(embeddings)
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        print("OV Duration:", duration, "ms", type(model_ov).__name__)
        return embeddings

def run_ovms():
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:6001/v3",api_key="unused"    )
    start_time = datetime.datetime.now()
    responses = client.embeddings.create(input=[text], model=model_id)
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    print("Duration:", duration, "ms - OVMS",)
    print(type(responses.data))
    for data in responses.data:
        print(type(data.embedding))
        result = np.array(data.embedding)
        print(result)

HF_embeddings = run_model(model_pt)
OV_embeddings = run_OV()
OVMS_embeddings = run_ovms()


