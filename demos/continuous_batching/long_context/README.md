# Long context optimizations {#ovms_demo_long_context}

Using models with very long context and prompts might be particularly challenging. The key goals are to get maximum throughput, minimal latency and reasonable memory consumption.
It is very common for applications using RAG chain, documents summarization, question answering and many more. 
Here is presented a set of optimization which can significantly boost performance :
- prefix caching
- prompt lookup
- tokens eviction

Prefix caching:
Prefix caching in large language models (LLMs) is an optimization technique used to improve performance when processing repeated or static parts of input prompts. Instead of recomputing the model's output for the same prefix (e.g., a fixed instruction or context), the results of the prefix are cached after the first computation. 
When the same prefix is encountered again, the cached results are reused, skipping redundant computations. This reduces latency and computational overhead, especially in scenarios like chatbots or applications with repetitive prompts.

Prompt lookup:
Prompt lookup in large language models (LLMs) is an optimization technique where precomputed responses or embeddings for frequently used prompts are stored in a database or cache. When the same prompt is encountered, the system retrieves the precomputed result instead of processing the prompt through the model again. This reduces latency and computational cost, especially in applications with repetitive or common queries.

Tokens eviction:
Tokens eviction in large language models (LLMs) is an optimization technique used to manage memory and computational efficiency when dealing with long input sequences. Since LLMs have a fixed context window (the maximum number of tokens they can process), tokens eviction involves removing or discarding older, less relevant tokens from the input sequence to make room for new ones.
This ensures that the model focuses on the most recent and relevant context while staying within its token limit. The algorithm typically prioritizes retaining tokens that are more critical for understanding the current context, such as recent dialogue or key instructions.

## Deployment

Let's demonstrate all the optimizations combined and test it with the real life scenario of sending multiple various questions in the same context. It will illustrate the gain from the prefix caching on the first token latency, improved second token latency thanks for prompt lookup and moderate memory consumption despite very long prompts and parallel execution.

Export the model Qwen/Qwen2.5-7B-Instruct-1 which has the max context length of 1 million tokens! 
```
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/requirements.txt
mkdir models
python export_model.py text_generation --source_model meta-llama/Llama-3.1-8B-Instruct  --config_file_path models/config.json --model_repository_path models --prompt_lookup --tokens_eviction --prefix_caching
```

Start OVMS:
```
docker run -d --rm -p 8000:8000 -v $(pwd)/models/meta-llama/Llama-3.1-8B-Instruct:/model:ro openvino/model_server:latest --rest_port 8000 --model_name meta-llama/Llama-3.1-8B --model_path /model
```

## Dataset for experiments

To test the performance using vllm benchmarking script, let's create a dataset with long shared context and a set of questions in each request. To make this experiment similar to real live, the context is not syntetic but build with the content of Don Quixote story.

```
python make_dataset.py --context_tokens 50000
```

It will create a file called `dataset.json`


## Testing performance

Let's check the performance 
```console
git clone --branch v0.6.0 --depth 1 https://github.com/vllm-project/vllm
cd vllm
pip3 install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd benchmarks
python benchmark_serving.py --host localhost --port 8000 --endpoint /v3/chat/completions --backend openai-chat --model meta-llama/Llama-3.1-8B-Instruct --dataset-path dataset.json --num-prompts 100 --request-rate inf -- extra_body '{"num_assistant_tokens" : 5, "max_ngram_size": 3 }'
```

The results shown above, despite very long context, have much lower TTFT latency with prefix caching. As long as the beginning of the request prompt is reused, KV cache can be also reused to speed up prompt processing.

## Testing accuracy

```bash
lm-eval --model local-chat-completions --tasks longbench_gov_report --model_args model=meta-llama/Llama-3.1-8B-Instruct,base_url=http://ov-spr-28.sclab.intel.com:8000/v3/chat/completions,num_concurrent=10,max_retries=3,tokenized_requests=False,timeout=1800  --verbosity DEBUG  --log_samples --output_path test/ --seed 1 --apply_chat_template --limit 100
[42:23<00:00, 25.43s/it]
|       Tasks        |Version|Filter|n-shot|  Metric   |   |Value |   |Stderr|
|--------------------|------:|------|-----:|-----------|---|-----:|---|-----:|
|longbench_gov_report|      2|none  |     0|rouge_score|↑  |0.3389|±  |0.0052|
```

Similar experiment can compare the results on cuda and expensive GPU card with 40GB
```
lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct --tasks longbench_gov_report --device cuda:0 --batch_size 1 --limit 
[35:13<00:00, 21.14s/it]
|       Tasks        |Version|Filter|n-shot|  Metric   |   |Value |   |Stderr|
|--------------------|------:|------|-----:|-----------|---|-----:|---|-----:|
|longbench_gov_report|      2|none  |     0|rouge_score|↑  |0.3418|±  |0.0071|
```

This test shows that despite model quantization, KV cache compression and tokens eviction mechanism, the model accuracy is not impacted significantly. The memory consumption is on a reasonable level while running this load on OVMS. Moreover, host RAM on the servers is generally much more available than VRAM on GPU cards. That make Xeon server suitable to running the models with long prompt in higher concurrency.


## Intel GPU considerations

Models with long context can be also successfully used on Intel GPUs and take advantage of the same optimizations. Because of the limitations of the VRAM, there are just a couple of recommendations to follow.

Export the model in INT4 precision

Running the load in high concurrency and long context, is recommended when the requests share the beginning of the prompt. That optimizes performance and allows sharing the KV cache

When running the load with completely different long context for each request, reduce the concurrency level.

