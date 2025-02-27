# Measuring Performance

Use [vllm benchmarking scripts](https://github.com/vllm-project/vllm) - vision arena bench dataset (with images).

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


