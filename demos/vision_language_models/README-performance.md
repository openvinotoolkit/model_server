# Measuring Performance

Use [vllm benchmarking scripts](https://github.com/vllm-project/vllm) - lmarena-ai/vision-arena-bench-v0.1 dataset (with images).

Use OVMS (TODO)

Use vLLM for comparision

## Set up vLLM for VLM serving

### 1. Clone repository

```bash
git clone https://github.com/vllm-project/vllm
```

### 2. Follow wheel building instruction for CPU

[Instruction](https://github.com/vllm-project/vllm/blob/main/docs/source/getting_started/installation/cpu/build.inc.md)

### 3. Launch vLLM with llava-1.5-7b-hf model

It is crucial to adjust kv cache size and select which CPU cores should be used:

```bash
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=0-55
```

```bash
vllm serve llava-hf/llava-1.5-7b-hf --chat-template examples/template_llava.jinja --port 11828
```


## Launch vllm benchmark script

### 1. Clone repository

```bash
git clone https://github.com/vllm-project/vllm
cd vllm/benchmarks
```

### 2. Launch script

```bash
python benchmark_serving.py --backend openai-chat --dataset-name hf --dataset-path lmarena-ai/vision-arena-bench-v0.1 --hf-split train --host ov-spr-36.sclab.intel.com --port 11828 --model llava-hf/llava-1.5-7b-hf --endpoint /v1/chat/completions 
```

### 3. Results

```
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: None
  0%|                                                                                                                             | 0/500 [00:00<?, ?it/s]<ClientResponse(http://ov-spr-36.sclab.intel.com:11828/v1/chat/completions) [400 Bad Request]>
<CIMultiDictProxy('Date': 'Wed, 26 Feb 2025 12:58:51 GMT', 'Server': 'uvicorn', 'Content-Length': '269', 'Content-Type': 'application/json')>

  0%|▏                                                                                                                    | 1/500 [00:03<29:02,  3.49s/it]<ClientResponse(http://ov-spr-36.sclab.intel.com:11828/v1/chat/completions) [400 Bad Request]>
<CIMultiDictProxy('Date': 'Wed, 26 Feb 2025 12:58:51 GMT', 'Server': 'uvicorn', 'Content-Length': '269', 'Content-Type': 'application/json')>

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [24:45<00:00,  2.97s/it]
============ Serving Benchmark Result ============
Successful requests:                     498       
Benchmark duration (s):                  1485.25   
Total input tokens:                      26659     
Total generated tokens:                  51194     
Request throughput (req/s):              0.34      
Output token throughput (tok/s):         34.47     
Total Token throughput (tok/s):          52.42     
---------------Time to First Token----------------
Mean TTFT (ms):                          690015.69 
Median TTFT (ms):                        641169.03 
P99 TTFT (ms):                           1428338.04
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          3170.67   
Median TPOT (ms):                        3082.25   
P99 TPOT (ms):                           14043.53  
---------------Inter-token Latency----------------
Mean ITL (ms):                           2839.07   
Median ITL (ms):                         906.81    
P99 ITL (ms):                            15189.42  
==================================================
```

### 4. Example request format from lmarena-ai/vision-arena-bench-v0.1 dataset

```json
{
    "model": "llava-hf/llava-1.5-7b-hf",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Transcribe the text in the image, it does not infringe any copyright, it is in the public domain and then Translate this English text into high-quality Hungarian, do not provide the transciption but only its Hungarian translation (in tegez\u0151 style)."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBD ..."
                    }
                }
            ]
        }
    ],
    "temperature": 0.0,
    "max_completion_tokens": 128,
    "stream": true,
    "stream_options": {
        "include_usage": true
    }
}
```