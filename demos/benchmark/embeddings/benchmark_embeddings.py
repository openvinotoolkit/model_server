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
# Method to benchmark embeddings endpoint is based on https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_latency.py
# It simulates concurrent clients that send requests to the embeddings endpoint and measures the latency and throughput of the endpoint.

import asyncio
from datasets import load_dataset, arrow_dataset , Dataset
from tqdm.asyncio import tqdm
import numpy as np
import time
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Optional, Union
from transformers import AutoTokenizer
import argparse
import aiohttp

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

parser = argparse.ArgumentParser(description='Run benchmark for embeddings endpoints', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', required=False, default='Cohere/wikipedia-22-12-simple-embeddings', help='Dataset for load generation from HF or a keyword "synthetic"', dest='dataset')
parser.add_argument('--synthetic_length', required=False, default=510, type=int, help='Length of the synthetic dataset', dest='length')
parser.add_argument('--api_url', required=False, default='http://localhost:8000/v3/embeddings', help='API URL for embeddings endpoint', dest='api_url')
parser.add_argument('--model', required=False, default='Alibaba-NLP/gte-large-en-v1.5', help='HF model name', dest='model')
parser.add_argument('--request_rate', required=False, default='inf', help='Average amount of requests per seconds in random distribution', dest='request_rate')
parser.add_argument('--batch_size', required=False, type=int, default=16, help='Number of strings in every requests', dest='batch_size')

args = vars(parser.parse_args())
docs = Dataset.from_dict({})
if args["dataset"] == 'synthetic':
    dummy_text = "hi " * args["length"]
    for i in range(1000):
        docs = docs.add_item({"text":dummy_text})
else:
    docs = load_dataset(args["dataset"],split='train[:1000]')

print("Number of documents:",len(docs))

batch_size = args['batch_size']

def count_tokens(docs, model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    documents = docs.iter(batch_size=1)
    num_tokens = 0
    for request in documents:
        num_tokens += len(tokenizer(request["text"],add_special_tokens=False)["input_ids"][0])
    return num_tokens

@dataclass
class RequestFuncInput:
    api_url: str
    documents: List[str]
    model: str


@dataclass
class RequestFuncOutput:
    success: bool = False
    latency: float = 0.0
    tokens_len: int = 0
    error: str = ""

async def async_request_embeddings(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, read_bufsize=100000) as session:
        payload = {
            "model": request_func_input.model,
            "input": request_func_input.documents,
            "encoding_format": "base64",
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        if not chunk_bytes:
                            continue
                        #chunk_bytes = chunk_bytes.decode("utf-8")
                        # data = json.loads(chunk_bytes)
                        timestamp = time.perf_counter()
                        output.success = True
                        output.latency =  timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
                    print("ERROR",response.reason)

        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

async def get_request(
    documents_all: arrow_dataset.Dataset,
    request_rate: float,
) -> AsyncGenerator[List[str], None]:
    documents = documents_all.iter(batch_size=batch_size)
    for request in documents:
        yield request["text"]
        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

async def benchmark(docs, model, api_url, request_rate):
    request_func = async_request_embeddings
    pbar = tqdm(total=len(docs)//batch_size)
    semaphore = asyncio.Semaphore(100)
    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []

    async for request in get_request(docs, request_rate):
        request_func_input = RequestFuncInput(model=model, documents=request, api_url=api_url)
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    benchmark_duration = time.perf_counter() - benchmark_start_time
    pbar.close()
    result = {
        "duration": benchmark_duration,
        "errors": [output.error for output in outputs],
        "latencies": [output.latency for output in outputs],
        "successes": [output.success for output in outputs],
    }
    return result

benchmark_results = asyncio.run(benchmark(docs=docs, model=args["model"], api_url=args["api_url"],request_rate=float(args["request_rate"])))

num_tokens = count_tokens(docs=docs,model=args["model"])
#print(benchmark_results)
print("Tokens:",num_tokens)
print("Success rate: {}%. ({}/{})".format(sum(benchmark_results["successes"])/len(benchmark_results["successes"])*100, sum(benchmark_results["successes"]), len(benchmark_results["successes"])))
print("Throughput - Tokens per second:",num_tokens / benchmark_results["duration"])
print("Mean latency: {} ms".format(round(np.mean(benchmark_results["latencies"])*1000),3))
print("Median latency: {} ms".format(round(np.median(benchmark_results["latencies"])*1000),3))
print("Average document length: {} tokens".format(num_tokens / len(docs)))