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
import base64
import io
import mimetypes
import secrets
from pathlib import Path
from urllib.parse import urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

default_url_description = "Default value depends on the backend: \
    ovms-embeddings: http://localhost:8000/v3/embeddings ;\
    ovms_rerank: http://localhost:8000/v3/rerank ;\
    tei_embed: http://localhost:8080/embed ;\
    infinity-embeddings: http://localhost:7997/embeddings ;\
    text2speech: http://localhost:8000/v3/audio/speech ;\
    speech2text: http://localhost:8000/v3/audio/transcriptions ;\
    translations: http://localhost:8000/v3/audio/translations ;\
    ovms-vlm: http://localhost:8000/v1/chat/completions"


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number

parser = argparse.ArgumentParser(description='Run benchmark for embeddings, audio, and VLM chat endpoints', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', required=False, default='Cohere/wikipedia-2023-11-embed-multilingual-v3', help='Dataset for load generation from HF or a keyword "synthetic"', dest='dataset')
parser.add_argument('--synthetic_length', required=False, default=510, type=int, help='Length of the synthetic dataset', dest='length')
parser.add_argument('--api_url', required=False, help='API URL for embeddings endpoint. ' + default_url_description, dest='api_url')
parser.add_argument('--model', required=False, help='Served model name; auto-detected from /v1/models when exactly one model is available', dest='model')
parser.add_argument('--request_rate', required=False, default='inf', help='Average amount of requests per seconds in random distribution', dest='request_rate')
parser.add_argument('--batch_size', required=False, type=positive_int, default=None, help='Strings per request; default 16, or 1 for ovms-vlm', dest='batch_size')
parser.add_argument('--voice', required=False, help='Voice name for text2speech backend', dest='voice')
parser.add_argument('--backend', required=False, default='ovms-embeddings', choices=['ovms-embeddings','tei-embed','infinity-embeddings','ovms_rerank','text2speech','speech2text', 'translations', 'ovms-vlm'], help='Backend serving API type', dest='backend')
parser.add_argument('--limit', required=False, type=positive_int, default=1000, help='Number of documents, or independent conversations for ovms-vlm', dest='limit')
parser.add_argument('--split', required=False, default='train', help='Dataset split', dest='split')
parser.add_argument('--hf-subset', required=False, help='Hf dataset subset', dest='subset')
parser.add_argument('--trust-remote-code', required=False, type=bool, default=False, help='Trust remote code from huggingface', dest='trust_remote_code')
parser.add_argument('--tokenizer', required=False, help='Optional local/HF tokenizer override; OVMS uses /v1/tokenize by default', dest='tokenizer')
parser.add_argument('--tokenize_url', help='Override the OVMS tokenize URL; default is /v1/tokenize on the API host')
parser.add_argument('--input_length', type=positive_int, default=510, help='First VLM user message text tokens; maximum when using --text_input_file')
parser.add_argument('--text_input_file', help='UTF-8 file for the first VLM user message, truncated to --input_length tokens')
parser.add_argument('--output_lenght', '--output_length', dest='output_length', type=positive_int, default=128, help='Maximum output tokens per VLM turn; sends ignore_eos=true')
parser.add_argument('--max_concurrency', type=positive_int, default=100, help='Maximum simultaneous requests; VLM conversations run sequential turns within this limit')
parser.add_argument('--image', help='Image HTTP(S) URL, data URI, or local filename for the first VLM user message')
parser.add_argument('--turns', type=positive_int, default=1, help='Requests per VLM conversation, including the first turn')
parser.add_argument('--turn_lenght', '--turn_length', dest='turn_length', type=positive_int, default=128, help='New text tokens added to each subsequent VLM user turn')

args = {}

backend_function = None
default_api_url = None

batch_size = 16


def synthetic_token_text(tokenizer, length):
    for seed in (' hello', ' a', ' test'):
        tokens = tokenizer.encode(seed, add_special_tokens=False)
        if not tokens:
            continue
        tokens = (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        if len(tokenizer.encode(text, add_special_tokens=False)) == length:
            return text
    raise ValueError(f'Could not generate exactly {length} text tokens with this tokenizer')


def uses_server_tokenization():
    return args['backend'] in ('ovms-vlm', 'ovms-embeddings', 'ovms_rerank') and not args.get('tokenizer')


def discover_model(api_url):
    address = urlsplit(api_url)
    endpoint = urlunsplit((address.scheme, address.netloc, '/v1/models', '', ''))
    request = Request(endpoint, headers=application_json_headers)
    with urlopen(request, timeout=60) as response:
        data = json.load(response)
    if not isinstance(data, dict) or not isinstance(data.get('data'), list):
        raise ValueError('Invalid /v1/models response; specify --model explicitly')
    models = []
    for model in data['data']:
        if not isinstance(model, dict) or not isinstance(model.get('id'), str) or not model['id']:
            raise ValueError('Invalid model ID in /v1/models response; specify --model explicitly')
        models.append(model['id'])
    if len(models) > 1:
        raise ValueError(f"Multiple models available; --model is required. Available models: {', '.join(models)}")
    if not models:
        raise ValueError('No models returned by /v1/models; specify --model explicitly')
    return models[0]


def tokenize_texts(texts, model):
    address = urlsplit(args['api_url'])
    endpoint = args.get('tokenize_url') or urlunsplit((address.scheme, address.netloc, '/v1/tokenize', '', ''))
    payload = {'model': model, 'text': texts, 'add_special_tokens': False}
    request = Request(endpoint, data=json.dumps(payload).encode('utf-8'), headers=application_json_headers)
    with urlopen(request, timeout=60) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise ValueError('Invalid /v1/tokenize response: expected an object')
    tokens = data.get('tokens')
    is_batch = isinstance(texts, list)
    rows = tokens if is_batch else [tokens]
    expected_rows = len(texts) if is_batch else 1
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ValueError('Invalid /v1/tokenize response: unexpected batch size')
    if any(not isinstance(row, list) or any(isinstance(token, bool) or not isinstance(token, int) or token < 0
                                          for token in row) for row in rows):
        raise ValueError('Invalid /v1/tokenize response: expected token IDs')
    return rows


def synthetic_server_token_text(length):
    for seed in (' hello', ' a', ' test'):
        low, high = 1, length
        while low <= high:
            repetitions = (low + high) // 2
            text = seed * repetitions
            count = len(tokenize_texts(text, args['model'])[0])
            if count == length:
                return text
            if count < length:
                low = repetitions + 1
            else:
                high = repetitions - 1
    raise ValueError(f'Could not generate exactly {length} text tokens using /v1/tokenize')


def truncate_input_text(text, length, token_count):
    if not text or token_count(text) <= length:
        return text
    prefix = ''
    low, high = 1, len(text) - 1
    while low <= high:
        end = (low + high) // 2
        candidate = text[:end]
        if token_count(candidate) <= length:
            prefix = candidate
            low = end + 1
        else:
            high = end - 1
    return prefix


def unique_initial_text(text, length, token_count, used):
    for attempt in range(32):
        candidate = f'{secrets.token_hex(16)} {text}'
        candidate = truncate_input_text(candidate, length, token_count)
        if candidate and candidate not in used:
            used.add(candidate)
            return candidate
    raise ValueError('Could not create distinct initial prompts; increase --input_length')


def image_context_url(image):
    if urlparse(image).scheme in ('http', 'https') or image.startswith('data:image/'):
        return image
    path = Path(image)
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type is None or not media_type.startswith('image/'):
        raise ValueError(f'Cannot determine image MIME type for {image}')
    return f'data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode("ascii")}'


def prepare_documents():
    if args['backend'] == 'ovms-vlm':
        source = Path(args['text_input_file']).read_text(encoding='utf-8') if args.get('text_input_file') else None
        if uses_server_tokenization():
            token_count = lambda value: len(tokenize_texts(value, args['model'])[0])
            text = (synthetic_server_token_text(args['input_length']) if source is None else
                    truncate_input_text(source, args['input_length'], token_count))
            args['turn_text'] = synthetic_server_token_text(args['turn_length'])
        else:
            tokenizer = AutoTokenizer.from_pretrained(args['tokenizer'] or args['model'],
                                                     trust_remote_code=args['trust_remote_code'])
            token_count = lambda value: len(tokenizer.encode(value, add_special_tokens=False))
            text = (synthetic_token_text(tokenizer, args['input_length']) if source is None else
                    truncate_input_text(source, args['input_length'], token_count))
            args['turn_text'] = synthetic_token_text(tokenizer, args['turn_length'])
        args['image_url'] = image_context_url(args['image']) if args['image'] else None
        used = set()
        texts = [unique_initial_text(text, args['input_length'], token_count, used)
                  for index in range(args['limit'])]
        return Dataset.from_dict({'text': texts})
    docs = Dataset.from_dict({})
    if args['dataset'] == 'synthetic':
        dummy_text = 'hi ' * args['length']
        for index in range(args['limit']):
            docs = docs.add_item({'text': dummy_text})
        return docs
    split = f"{args['split']}[:{args['limit']}]"
    if args['subset'] is None:
        return load_dataset(args['dataset'], trust_remote_code=args['trust_remote_code'], split=split)
    return load_dataset(args['dataset'], args['subset'], trust_remote_code=args['trust_remote_code'], split=split)

def count_tokens(docs, model):
    if uses_server_tokenization():
        return sum(len(tokens) for request in docs.iter(batch_size=64)
                   for tokens in tokenize_texts(request['text'], model))
    if args["tokenizer"] == None:
        hf_tokenizer = model
    else:
        hf_tokenizer = args["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
    documents = docs.iter(batch_size=1)
    num_tokens = 0
    for request in documents:
        num_tokens += len(tokenizer(request["text"],add_special_tokens=False, truncation=True)["input_ids"][0])
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
    audio_duration: float = 0.0
    error: str = ""
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    ttft: Optional[float] = None
    tpot: Optional[float] = None
    inter_chunk_latencies: List[float] = field(default_factory=list)

application_json_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

application_multipart_headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }


def read_vlm_usage(data):
    usage = data['usage']
    prompt = usage['prompt_tokens']
    completion = usage['completion_tokens']
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
    for value in (prompt, completion, cached):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('Usage token counts must be non-negative integers')
    if cached > prompt:
        raise ValueError('Cached tokens cannot exceed prompt tokens')
    return prompt, completion, cached


async def sse_events(content):
    data_lines = []
    async for raw_line in content:
        line = raw_line.decode('utf-8').rstrip('\r\n')
        if not line:
            if data_lines:
                yield '\n'.join(data_lines), time.perf_counter()
                data_lines = []
        elif line.startswith('data:'):
            data_lines.append(line[5:].lstrip(' '))
    if data_lines:
        yield '\n'.join(data_lines), time.perf_counter()


async def read_vlm_stream(response, output, started):
    fragments = []
    arrivals = []
    usage = None
    done = False
    async for event, arrived in sse_events(response.content):
        if event.strip() == '[DONE]':
            done = True
            break
        data = json.loads(event)
        if data.get('error'):
            raise ValueError(f"Streaming error: {data['error']}")
        if data.get('usage') is not None:
            usage = data
        generated = False
        for choice in data.get('choices') or []:
            if choice.get('index', 0) != 0:
                continue
            delta = choice.get('delta') or {}
            text = delta.get('content') or ''
            reasoning = delta.get('reasoning_content') or ''
            if not isinstance(text, str) or not isinstance(reasoning, str):
                raise ValueError('Expected streamed text content')
            if text:
                fragments.append(text)
            generated = generated or bool(text or reasoning)
        if generated:
            arrivals.append(arrived)
    if not done:
        raise ValueError('Incomplete stream: missing [DONE]')
    if usage is None:
        raise ValueError('Missing usage in streaming response')
    output.input_tokens, output.output_tokens, output.cached_tokens = read_vlm_usage(usage)
    output.text = ''.join(fragments)
    if arrivals:
        output.ttft = arrivals[0] - started
        output.inter_chunk_latencies = [current - previous for previous, current in zip(arrivals, arrivals[1:])]
    if arrivals and output.output_tokens > 0:
        output.tpot = (arrivals[-1] - arrivals[0]) / output.output_tokens


async def async_request_vlm(request_func_input: RequestFuncInput, pbar=None):
    content = [{'type': 'text', 'text': request_func_input.documents[0]}]
    if args['image_url']:
        content.append({'type': 'image_url', 'image_url': {'url': args['image_url']}})
    messages = [{'role': 'user', 'content': content}]
    outputs = []
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, read_bufsize=100000) as session:
        for turn in range(args['turns']):
            if turn:
                messages.append({'role': 'user', 'content': args['turn_text']})
            payload = {
                'model': request_func_input.model,
                'messages': messages,
                'max_completion_tokens': args['output_length'],
                'ignore_eos': True,
                'stream': True,
                'stream_options': {'include_usage': True},
            }
            output = RequestFuncOutput()
            started = time.perf_counter()
            try:
                async with session.post(request_func_input.api_url, json=payload,
                                        headers=application_json_headers) as response:
                    if response.status != 200:
                        raise ValueError(f'HTTP {response.status}: {(await response.text())[:1000]}')
                    await read_vlm_stream(response, output, started)
                output.success = True
                messages.append({'role': 'assistant', 'content': output.text})
            except Exception as error:
                output.error = str(error)
            output.latency = time.perf_counter() - started
            outputs.append(output)
            if pbar:
                pbar.update(1)
            if not output.success:
                break
    return outputs


async def async_request_text2speech(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    import soundfile
    api_url = request_func_input.api_url

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, read_bufsize=100000) as session:
        payload = {
            "model": request_func_input.model,
            "input": request_func_input.documents[0],
        }
        if args["voice"] is not None:
            payload["voice"] = args["voice"]
        headers = application_json_headers

        output = RequestFuncOutput()
        st = time.perf_counter()
        audio_bytes = bytearray()
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        if not chunk_bytes:
                            continue
                        audio_bytes.extend(chunk_bytes)
                    output.success = True
                    output.latency = time.perf_counter() - st
                    try:
                        output.audio_duration = soundfile.info(io.BytesIO(audio_bytes)).duration
                    except Exception as e:
                        output.success = False
                        output.error = f"Could not decode WAV response to compute audio duration: {e}"
                else:
                    output.error = response.reason or ""
                    output.success = False
                    print("ERROR", response.reason)

        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

async def async_request_speech2text(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    import soundfile
    api_url = request_func_input.api_url

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, read_bufsize=100000) as session:
        headers = application_multipart_headers

        y, sr = request_func_input.documents[0]["array"], request_func_input.documents[0]["sampling_rate"]
        buffer = io.BytesIO()
        soundfile.write(buffer, y, sr, format="WAV")
        buffer.seek(0)

        form = aiohttp.FormData()
        form.add_field('file', buffer, content_type='audio/wav')
        form.add_field('model', request_func_input.model)
        output = RequestFuncOutput()
        st = time.perf_counter()
        try:
            async with session.post(url=api_url, data=form,
                                    headers=headers) as response:                
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        if not chunk_bytes:
                            continue
                        timestamp = time.perf_counter()
                        output.success = True
                        output.latency =  timestamp - st
                        output.text = chunk_bytes.decode("utf-8")
                else:
                    output.error = response.reason or ""
                    output.success = False

        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

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
        headers = application_json_headers

        output = RequestFuncOutput()
        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        if not chunk_bytes:
                            continue
                        # uncomment for response debugging
                        # chunk_bytes = chunk_bytes.decode("utf-8")
                        # data = json.loads(chunk_bytes)
                        # TBD: saving response to file
                        timestamp = time.perf_counter()
                        output.success = True
                        output.latency =  timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
                    print("ERROR", response.reason)

        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

async def async_request_rerank(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, read_bufsize=100000) as session:
        payload = {
            "model": request_func_input.model,
            "documents": request_func_input.documents,
            "query": "Hello"
        }
        headers = application_json_headers

        output = RequestFuncOutput()
        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        if not chunk_bytes:
                            continue
                        # uncomment for response debugging
                        # chunk_bytes = chunk_bytes.decode("utf-8")
                        # data = json.loads(chunk_bytes)
                        # TBD: saving response to file
                        timestamp = time.perf_counter()
                        output.success = True
                        output.latency =  timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
                    print("ERROR", response.reason)

        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

async def async_request_embeddings_tei(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, read_bufsize=100000) as session:
        payload = {
            "inputs": request_func_input.documents,
        }
        headers = application_json_headers

        output = RequestFuncOutput()
        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        if not chunk_bytes:
                            continue
                        # uncomment for response debugging
                        # chunk_bytes = chunk_bytes.decode("utf-8")
                        # data = json.loads(chunk_bytes)
                        # TBD: saving response to file
                        timestamp = time.perf_counter()
                        output.success = True
                        output.latency =  timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
                    print("ERROR", response.reason)

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
        if args["backend"] == "speech2text" or args["backend"] == "translations":
            yield request["audio"]
        else:
            yield request["text"]
        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

async def benchmark(docs, model, api_url, request_rate, backend_function):
    request_func = backend_function
    request_count = len(docs)//batch_size + (len(docs) % batch_size > 0)
    pbar = tqdm(total=request_count * (args['turns'] if args['backend'] == 'ovms-vlm' else 1))
    semaphore = asyncio.Semaphore(args['max_concurrency'])
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
    if args['backend'] == 'ovms-vlm':
        outputs = [output for conversation in outputs for output in conversation]
    benchmark_duration = time.perf_counter() - benchmark_start_time
    pbar.close()
    if args["backend"] == "speech2text" or args["backend"] == "translations":
        if args["tokenizer"] == None:
            hf_tokenizer = model
        else:
            hf_tokenizer = args["tokenizer"]
        tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
        for output in outputs:
            data = json.loads(output.text)
            output.tokens_len =  len(tokenizer(data['text'],add_special_tokens=False, truncation=True)["input_ids"])

    result = {
        "duration": benchmark_duration,
        "errors": [output.error for output in outputs],
        "latencies": [output.latency for output in outputs],
        "successes": [output.success for output in outputs],
        "token_count": [output.tokens_len for output in outputs],
        "audio_durations": [output.audio_duration for output in outputs],
        "input_tokens": [output.input_tokens for output in outputs],
        "output_tokens": [output.output_tokens for output in outputs],
        "cached_tokens": [output.cached_tokens for output in outputs],
        "ttfts": [output.ttft for output in outputs],
        "tpots": [output.tpot for output in outputs],
        "inter_chunk_latencies": [output.inter_chunk_latencies for output in outputs],
    }
    return result

def print_vlm_statistics(results):
    successes = results['successes']
    successful_count = sum(successes)
    attempted = len(successes)
    duration = results['duration']
    prompt = sum(value for value, success in zip(results['input_tokens'], successes) if success)
    completion = sum(value for value, success in zip(results['output_tokens'], successes) if success)
    cached = sum(value for value, success in zip(results['cached_tokens'], successes) if success)
    print(f"Planned requests: {args['limit'] * args['turns']} ({args['limit']} conversations, {args['turns']} turns)")
    print(f'Number of requests: {attempted}')
    print(f'Success rate: {100 * successful_count / attempted if attempted else 0:.2f}%. ({successful_count}/{attempted})')
    print(f'Input tokens (including cached): {prompt}')
    print(f'Output tokens: {completion}')
    print(f'Cached input tokens: {cached}')
    print(f'Uncached input tokens: {prompt - cached}')
    print(f'Cache hit rate: {100 * cached / prompt if prompt else 0:.2f}%')
    print(f'Elapsed time: {duration:.3f} s')
    for label, count in (('Input tokens (including cached)', prompt),
                         ('Uncached input tokens', prompt - cached), ('Output tokens', completion)):
        print(f'Throughput - {label} per second: {count / duration if duration else 0:.3f}')
    print(f'Throughput - Successful requests per second: {successful_count / duration if duration else 0:.3f}')
    latencies = [value * 1000 for value, success in zip(results['latencies'], successes) if success]
    for label, operation in (('Mean', np.mean), ('Median', np.median),
                             ('p90', lambda values: np.percentile(values, 90)),
                             ('p99', lambda values: np.percentile(values, 99)), ('Stddev', np.std)):
        value = f'{operation(latencies):.3f} ms' if latencies else 'n/a'
        print(f'{label} latency: {value}')
    for metric, samples in (
        ('TTFT', [value * 1000 for value, success in zip(results.get('ttfts', []), successes)
                  if success and value is not None]),
        ('TPOT', [value * 1000 for value, success in zip(results.get('tpots', []), successes)
                  if success and value is not None]),
        ('inter-chunk latency', [value * 1000 for values, success in
                                 zip(results.get('inter_chunk_latencies', []), successes)
                                 if success for value in values]),
    ):
        for label, operation in (('Mean', np.mean), ('Median', np.median)):
            value = f'{operation(samples):.3f} ms' if samples else 'n/a'
            print(f'{label} {metric}: {value}')
    for error in list(dict.fromkeys(error for error in results['errors'] if error))[:3]:
        print(f'Request error: {error}', file=sys.stderr)


def main(argv=None):
    global args, batch_size, backend_function, default_api_url
    args = vars(parser.parse_args(argv))
    if args['text_input_file'] and args['backend'] != 'ovms-vlm':
        parser.error('--text_input_file is only supported for ovms-vlm')
    args['batch_size'] = args['batch_size'] or (1 if args['backend'] == 'ovms-vlm' else 16)
    batch_size = args['batch_size']
    if args['backend'] in ('ovms-vlm', 'text2speech', 'speech2text', 'translations') and batch_size != 1:
        parser.error(f"Only batch_size=1 is supported for {args['backend']}")
    try:
        request_rate = float(args['request_rate'])
    except ValueError:
        parser.error('--request_rate must be positive or inf')
    if np.isnan(request_rate) or request_rate <= 0:
        parser.error('--request_rate must be positive or inf')
    backends = {
        'ovms-embeddings': (async_request_embeddings, 'http://localhost:8000/v3/embeddings'),
        'ovms_rerank': (async_request_rerank, 'http://localhost:8000/v3/rerank'),
        'tei-embed': (async_request_embeddings_tei, 'http://localhost:8080/embed'),
        'infinity-embeddings': (async_request_embeddings, 'http://localhost:7997/embeddings'),
        'text2speech': (async_request_text2speech, 'http://localhost:8000/v3/audio/speech'),
        'speech2text': (async_request_speech2text, 'http://localhost:8000/v3/audio/transcriptions'),
        'translations': (async_request_speech2text, 'http://localhost:8000/v3/audio/translations'),
        'ovms-vlm': (async_request_vlm, 'http://localhost:8000/v1/chat/completions'),
    }
    backend_function, default_api_url = backends[args['backend']]
    if args['api_url'] is None:
        args['api_url'] = default_api_url
    try:
        if args['model'] is None:
            args['model'] = discover_model(args['api_url'])
            print(f"Selected model: {args['model']}")
        docs = prepare_documents()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print('Number of documents:', len(docs))
    benchmark_results = asyncio.run(benchmark(docs=docs, model=args['model'], api_url=args['api_url'],
                                            request_rate=request_rate, backend_function=backend_function))
    success_count = sum(benchmark_results['successes'])
    total_count = len(benchmark_results['successes'])
    success_rate = (success_count / total_count * 100.0) if total_count else 0.0

    if args['backend'] == 'ovms-vlm':
        print_vlm_statistics(benchmark_results)
        return 0 if all(benchmark_results['successes']) else 1
    if args['backend'] == 'text2speech':
        rtfx_values = []
        for latency, audio_duration, success in zip(
            benchmark_results['latencies'], benchmark_results['audio_durations'], benchmark_results['successes'],
        ):
            if success and latency > 0 and audio_duration > 0:
                rtfx_values.append(audio_duration / latency)
        print(f'Success rate: {success_rate}%. ({success_count}/{total_count})')
        print(f"Mean latency: {np.mean(benchmark_results['latencies'])*1000:.2f} ms")
        print(f"Median latency: {np.median(benchmark_results['latencies'])*1000:.2f} ms")
        if rtfx_values:
            print(f'Mean RTFx: {np.mean(rtfx_values):.3f}x')
            print(f'Median RTFx: {np.median(rtfx_values):.3f}x')
        else:
            print('Mean RTFx: n/a')
            print('Median RTFx: n/a')
    elif args['backend'] in ('speech2text', 'translations'):
        num_tokens = sum(benchmark_results['token_count'])
        print('Tokens:', num_tokens)
        print(f'Success rate: {success_rate}%. ({success_count}/{total_count})')
        print(f"Throughput - Tokens per second: {num_tokens / benchmark_results['duration']:^,.1f}")
        successful_latencies = [value for value, success in zip(benchmark_results['latencies'],
                                                                benchmark_results['successes']) if success and value > 0]
        if successful_latencies:
            print(f'Mean latency: {np.mean(successful_latencies)*1000:.2f} ms')
            print(f'Median latency: {np.median(successful_latencies)*1000:.2f} ms')
        else:
            print('Mean latency: n/a')
            print('Median latency: n/a')
        print(f'Average document length: {num_tokens / len(docs)} tokens')
    else:
        num_tokens = count_tokens(docs=docs, model=args['model'])
        print('Tokens:', num_tokens)
        print(f'Success rate: {success_rate}%. ({success_count}/{total_count})')
        print(f"Throughput - Tokens per second: {num_tokens / benchmark_results['duration']:^,.1f}")
        print(f"Mean latency: {np.mean(benchmark_results['latencies'])*1000:.2f} ms")
        print(f"Median latency: {np.median(benchmark_results['latencies'])*1000:.2f} ms")
        print(f'Average document length: {num_tokens / len(docs)} tokens')
    return 0


if __name__ == '__main__':
    sys.exit(main())
