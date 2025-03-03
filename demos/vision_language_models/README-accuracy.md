# Validating Accuracy

Use [lmms-eval project](https://github.com/EvolvingLMMs-Lab/lmms-eval) - mme and mmmu_val tasks.

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
vllm serve llava-hf/llava-1.5-7b-hf --chat-template examples/template_llava.jinja --port 11828 --limit-mm-per-prompt image=32
```

> NOTE: vLLM by default limits max 1 image. Dataset contains examples with more than 1 image in a request. Use `--limit-mm-per-prompt image=32` to allow more images.


## Launch lmms-eval

### 1. Clone repository

### 2. Install dependencies

```bash
pip install lmms-eval openai
```

### 3. Export OpenAI credentials, point to local server

```bash
export OPENAI_COMPATIBLE_API_URL=http://<address>:<port>/v1
export OPENAI_COMPATIBLE_API_KEY="undefined"
```

> NOTE: Use `v3` for OVMS, `v1` for vLLM.

### 4. Launch evaluation

```bash
python -m lmms_eval \
    --model openai_compatible \
    --model_args model_version=llava-hf/llava-1.5-7b-hf,max_retries=1 \
    --tasks mme,mmmu_val \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix openai_compatible \
    --output_path ./logs \
    --limit=250
```

It will launch 250 multi modal requests per task (500 in total) 

### 5. Results

Results:
```
| Tasks  |Version|Filter|n-shot|       Metric       |   | Value |   |Stderr|
|--------|-------|------|-----:|--------------------|---|------:|---|------|
|mme     |Yaml   |none  |     0|mme_cognition_score |↑  |57.5000|±  |   N/A|
|mme     |Yaml   |none  |     0|mme_perception_score|↑  |91.4286|±  |   N/A|
|mmmu_val|      0|none  |     0|mmmu_acc            |↑  | 0.3160|±  |   N/A|
```

Intermediate logs:

```
2025-03-03 14:37:38.454 | INFO     | __main__:cli_evaluate:295 - Verbosity set to INFO
2025-03-03 14:37:40.899 | INFO     | __main__:cli_evaluate_single:378 - Evaluation tracker args: {'output_path': './logs'}
2025-03-03 14:37:40.899 | WARNING  | __main__:cli_evaluate_single:400 -  --limit SHOULD ONLY BE USED FOR TESTING.REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.
2025-03-03 14:37:40.900 | INFO     | __main__:cli_evaluate_single:467 - Selected Tasks: ['mme', 'mmmu_val']
2025-03-03 14:37:40.903 | INFO     | lmms_eval.evaluator:simple_evaluate:155 - Setting random seed to 0 | Setting numpy seed to 1234 | Setting torch manual seed to 1234
2025-03-03 14:37:48.676 | INFO     | lmms_eval.evaluator_utils:from_taskdict:91 - No metadata found in task config for mme, using default n_shot=0
2025-03-03 14:37:48.677 | INFO     | lmms_eval.api.task:build_all_requests:425 - Building contexts for mmmu_val on rank 0...
100%|█████████████████████████████████████████████████████| 250/250 [00:00<00:00, 8370.79it/s]
2025-03-03 14:37:48.708 | INFO     | lmms_eval.api.task:build_all_requests:425 - Building contexts for mme on rank 0...
100%|████████████████████████████████████████████████████| 250/250 [00:00<00:00, 16259.01it/s]
2025-03-03 14:37:48.724 | INFO     | lmms_eval.evaluator:evaluate:447 - Running generate_until requests
Model Responding: 100%|█████████████████████████████████████| 500/500 [29:33<00:00,  3.55s/it]
Postprocessing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:00<00:00, 3927.50it/s]
Postprocessing: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:00<00:00, 10504.14it/s]
{'Overall-Art and Design': {'num': 60, 'acc': 0.46667}, 'Art': {'num': 30, 'acc': 0.43333}, 'Art_Theory': {'num': 30, 'acc': 0.5}, 'Overall-Business': {'num': 30, 'acc': 0.3}, 'Accounting': {'num': 30, 'acc': 0.3}, 'Overall-Science': {'num': 60, 'acc': 0.26667}, 'Biology': {'num': 30, 'acc': 0.26667}, 'Chemistry': {'num': 30, 'acc': 0.26667}, 'Overall-Health and Medicine': {'num': 40, 'acc': 0.275}, 'Basic_Medical_Science': {'num': 30, 'acc': 0.26667}, 'Clinical_Medicine': {'num': 10, 'acc': 0.3}, 'Overall-Humanities and Social Science': {'num': 0, 'acc': 0}, 'Overall-Tech and Engineering': {'num': 60, 'acc': 0.25}, 'Agriculture': {'num': 30, 'acc': 0.33333}, 'Architecture_and_Engineering': {'num': 30, 'acc': 0.16667}, 'Overall': {'num': 250, 'acc': 0.316}}
2025-03-03 15:07:22.121 | INFO     | utils:mme_aggregate_results:124 - code_reasoning: 57.50
2025-03-03 15:07:22.121 | INFO     | utils:mme_aggregate_results:124 - artwork: 91.43
2025-03-03 15:07:22.164 | INFO     | lmms_eval.loggers.evaluation_tracker:save_results_aggregated:188 - Saving results aggregated
2025-03-03 15:07:22.168 | INFO     | lmms_eval.loggers.evaluation_tracker:save_results_samples:255 - Saving per-sample results for: mme
2025-03-03 15:07:22.185 | INFO     | lmms_eval.loggers.evaluation_tracker:save_results_samples:255 - Saving per-sample results for: mmmu_val
openai_compatible (model_version=llava-hf/llava-1.5-7b-hf,max_retries=1), gen_kwargs: (), limit: 250.0, num_fewshot: None, batch_size: 1
```
