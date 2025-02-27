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
vllm serve llava-hf/llava-1.5-7b-hf --chat-template examples/template_llava.jinja --port 11828
```


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

> NOTE: `llava-hf/llava-1.5-7b-h` does not support more than 1 image. Dataset contains such examples. Those will fail.

### 5. Results

Results:
```
| Tasks  |Version|Filter|n-shot|       Metric       |   | Value |   |Stderr|
|--------|-------|------|-----:|--------------------|---|------:|---|------|
|mme     |Yaml   |none  |     0|mme_cognition_score |↑  |57.5000|±  |   N/A|
|mme     |Yaml   |none  |     0|mme_perception_score|↑  |91.9048|±  |   N/A|
|mmmu_val|      0|none  |     0|mmmu_acc            |↑  | 0.2920|±  |   N/A|
```

Intermediate logs:

```
2025-02-27 14:09:01.485 | INFO     | __main__:cli_evaluate:295 - Verbosity set to INFO
2025-02-27 14:09:03.952 | INFO     | __main__:cli_evaluate_single:378 - Evaluation tracker args: {'output_path': './logs'}
2025-02-27 14:09:03.952 | WARNING  | __main__:cli_evaluate_single:400 -  --limit SHOULD ONLY BE USED FOR TESTING.REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.
2025-02-27 14:09:03.953 | INFO     | __main__:cli_evaluate_single:467 - Selected Tasks: ['mme', 'mmmu_val']
2025-02-27 14:09:03.954 | INFO     | lmms_eval.evaluator:simple_evaluate:155 - Setting random seed to 0 | Setting numpy seed to 1234 | Setting torch manual seed to 1234
2025-02-27 14:09:11.541 | INFO     | lmms_eval.evaluator_utils:from_taskdict:91 - No metadata found in task config for mme, using default n_shot=0
2025-02-27 14:09:11.541 | INFO     | lmms_eval.api.task:build_all_requests:425 - Building contexts for mmmu_val on rank 0...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:00<00:00, 7961.49it/s]
2025-02-27 14:09:11.574 | INFO     | lmms_eval.api.task:build_all_requests:425 - Building contexts for mme on rank 0...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:00<00:00, 15190.37it/s]
2025-02-27 14:09:11.591 | INFO     | lmms_eval.evaluator:evaluate:447 - Running generate_until requests
Model Responding:  17%|████████████████████████████▌                                                                                                                                               | 83/500 [07:44<36:42,  5.28s/it]2025-02-27 14:16:56.053 | INFO     | lmms_eval.models.openai_compatible:generate_until:200 - Attempt 1/1 failed with error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
2025-02-27 14:16:56.054 | ERROR    | lmms_eval.models.openai_compatible:generate_until:204 - All 1 attempts failed. Last error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
Model Responding:  24%|█████████████████████████████████████████                                                                                                                                  | 120/500 [10:26<25:56,  4.10s/it]2025-02-27 14:19:39.406 | INFO     | lmms_eval.models.openai_compatible:generate_until:200 - Attempt 1/1 failed with error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
2025-02-27 14:19:39.407 | ERROR    | lmms_eval.models.openai_compatible:generate_until:204 - All 1 attempts failed. Last error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
Model Responding:  25%|██████████████████████████████████████████                                                                                                                                 | 123/500 [10:35<22:56,  3.65s/it]2025-02-27 14:19:47.824 | INFO     | lmms_eval.models.openai_compatible:generate_until:200 - Attempt 1/1 failed with error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
2025-02-27 14:19:47.824 | ERROR    | lmms_eval.models.openai_compatible:generate_until:204 - All 1 attempts failed. Last error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
Model Responding:  26%|████████████████████████████████████████████▍                                                                                                                              | 130/500 [11:05<27:33,  4.47s/it]2025-02-27 14:20:17.650 | INFO     | lmms_eval.models.openai_compatible:generate_until:200 - Attempt 1/1 failed with error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
2025-02-27 14:20:17.650 | ERROR    | lmms_eval.models.openai_compatible:generate_until:204 - All 1 attempts failed. Last error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
Model Responding:  27%|██████████████████████████████████████████████▏                                                                                                                            | 135/500 [11:27<27:52,  4.58s/it]2025-02-27 14:20:39.292 | INFO     | lmms_eval.models.openai_compatible:generate_until:200 - Attempt 1/1 failed with error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
2025-02-27 14:20:39.293 | ERROR    | lmms_eval.models.openai_compatible:generate_until:204 - All 1 attempts failed. Last error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
Model Responding:  29%|█████████████████████████████████████████████████▉                                                                                                                         | 146/500 [12:12<25:00,  4.24s/it]2025-02-27 14:21:24.482 | INFO     | lmms_eval.models.openai_compatible:generate_until:200 - Attempt 1/1 failed with error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
2025-02-27 14:21:24.483 | ERROR    | lmms_eval.models.openai_compatible:generate_until:204 - All 1 attempts failed. Last error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
Model Responding:  41%|█████████████████████████████████████████████████████████████████████▍                                                                                                     | 203/500 [16:23<28:27,  5.75s/it]2025-02-27 14:25:35.107 | INFO     | lmms_eval.models.openai_compatible:generate_until:200 - Attempt 1/1 failed with error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
2025-02-27 14:25:35.108 | ERROR    | lmms_eval.models.openai_compatible:generate_until:204 - All 1 attempts failed. Last error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
Model Responding:  42%|███████████████████████████████████████████████████████████████████████▏                                                                                                   | 208/500 [16:41<21:18,  4.38s/it]2025-02-27 14:25:52.935 | INFO     | lmms_eval.models.openai_compatible:generate_until:200 - Attempt 1/1 failed with error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
2025-02-27 14:25:52.935 | ERROR    | lmms_eval.models.openai_compatible:generate_until:204 - All 1 attempts failed. Last error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
Model Responding:  44%|██████████████████████████████████████████████████████████████████████████▉                                                                                                | 219/500 [17:23<20:30,  4.38s/it]2025-02-27 14:26:35.138 | INFO     | lmms_eval.models.openai_compatible:generate_until:200 - Attempt 1/1 failed with error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
2025-02-27 14:26:35.138 | ERROR    | lmms_eval.models.openai_compatible:generate_until:204 - All 1 attempts failed. Last error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
Model Responding:  44%|███████████████████████████████████████████████████████████████████████████▌                                                                                               | 221/500 [17:28<16:51,  3.63s/it]2025-02-27 14:26:40.583 | INFO     | lmms_eval.models.openai_compatible:generate_until:200 - Attempt 1/1 failed with error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
2025-02-27 14:26:40.583 | ERROR    | lmms_eval.models.openai_compatible:generate_until:204 - All 1 attempts failed. Last error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
Model Responding:  45%|█████████████████████████████████████████████████████████████████████████████▋                                                                                             | 227/500 [17:49<16:39,  3.66s/it]2025-02-27 14:27:01.413 | INFO     | lmms_eval.models.openai_compatible:generate_until:200 - Attempt 1/1 failed with error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
2025-02-27 14:27:01.413 | ERROR    | lmms_eval.models.openai_compatible:generate_until:204 - All 1 attempts failed. Last error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
Model Responding:  47%|████████████████████████████████████████████████████████████████████████████████▎                                                                                          | 235/500 [18:20<18:33,  4.20s/it]2025-02-27 14:27:31.851 | INFO     | lmms_eval.models.openai_compatible:generate_until:200 - Attempt 1/1 failed with error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
2025-02-27 14:27:31.851 | ERROR    | lmms_eval.models.openai_compatible:generate_until:204 - All 1 attempts failed. Last error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
Model Responding:  48%|█████████████████████████████████████████████████████████████████████████████████▍                                                                                         | 238/500 [18:28<15:37,  3.58s/it]2025-02-27 14:27:40.555 | INFO     | lmms_eval.models.openai_compatible:generate_until:200 - Attempt 1/1 failed with error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
2025-02-27 14:27:40.555 | ERROR    | lmms_eval.models.openai_compatible:generate_until:204 - All 1 attempts failed. Last error: Error code: 400 - {'object': 'error', 'message': 'At most 1 image(s) may be provided in one request.', 'type': 'BadRequestError', 'param': None, 'code': 400}
Model Responding: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [36:22<00:00,  4.37s/it]
Postprocessing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:00<00:00, 4052.19it/s]
Postprocessing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [00:00<00:00, 11600.58it/s]
{'Overall-Art and Design': {'num': 60, 'acc': 0.4}, 'Art': {'num': 30, 'acc': 0.43333}, 'Art_Theory': {'num': 30, 'acc': 0.36667}, 'Overall-Business': {'num': 30, 'acc': 0.3}, 'Accounting': {'num': 30, 'acc': 0.3}, 'Overall-Science': {'num': 60, 'acc': 0.25}, 'Biology': {'num': 30, 'acc': 0.26667}, 'Chemistry': {'num': 30, 'acc': 0.23333}, 'Overall-Health and Medicine': {'num': 40, 'acc': 0.25}, 'Basic_Medical_Science': {'num': 30, 'acc': 0.23333}, 'Clinical_Medicine': {'num': 10, 'acc': 0.3}, 'Overall-Humanities and Social Science': {'num': 0, 'acc': 0}, 'Overall-Tech and Engineering': {'num': 60, 'acc': 0.25}, 'Agriculture': {'num': 30, 'acc': 0.33333}, 'Architecture_and_Engineering': {'num': 30, 'acc': 0.16667}, 'Overall': {'num': 250, 'acc': 0.292}}
2025-02-27 14:45:34.390 | INFO     | utils:mme_aggregate_results:124 - code_reasoning: 57.50
2025-02-27 14:45:34.390 | INFO     | utils:mme_aggregate_results:124 - artwork: 91.90
2025-02-27 14:45:34.425 | INFO     | lmms_eval.loggers.evaluation_tracker:save_results_aggregated:188 - Saving results aggregated
2025-02-27 14:45:34.428 | INFO     | lmms_eval.loggers.evaluation_tracker:save_results_samples:255 - Saving per-sample results for: mme
2025-02-27 14:45:34.445 | INFO     | lmms_eval.loggers.evaluation_tracker:save_results_samples:255 - Saving per-sample results for: mmmu_val
openai_compatible (model_version=llava-hf/llava-1.5-7b-hf,max_retries=1), gen_kwargs: (), limit: 250.0, num_fewshot: None, batch_size: 1
```
