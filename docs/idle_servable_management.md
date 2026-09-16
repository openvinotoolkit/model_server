# Idle Servable Management (Preview) {#ovms_docs_idle_servable_management}

> **Preview feature:** Idle servable management is disabled by default. Its behavior and configuration can change in future releases.

Idle servable management keeps configured models and MediaPipe graphs visible to clients while unloading their heavy runtime resources when they are not needed. A request for an unloaded servable loads it again before inference. This lets one OVMS instance serve several large models with limited CPU or GPU memory.

Optionally set an OpenVINO model cache directory with `--cache_dir` to reduce reload latency. This is recommended because reloading from cache avoids recompiling the model. The first request still bears the latency cost of waiting for the model to wake up.

## Enable idle management

Set `--idle_unload_timeout_seconds` to a positive number when starting OVMS. `0`, the default, disables idle servable management.

Start OVMS with a cache directory:

```text
ovms --config_path /models/config.json --cache_dir /models/cache --idle_unload_timeout_seconds 60
```

## Groups

A group is a unit that OVMS loads and unloads together. Add `group_name` to a servable classic model or MediaPipe graph configuration. All servables in the same group load and unload together. Use groups for servables that must be available together, such as a graph and its dependent models.

Without `group_name`, every classic model and MediaPipe graph is assigned to its own group. A group named `permanent` stays loaded and is never unloaded.

For groups that contain only one non-permanent model or graph, setting `group_name` is optional. Keeping it explicit in configuration can improve readability.

OVMS keeps one non-permanent group active. A request for servable from another group waits for current group to finish its requests, then loads requested group. First request after unload includes wake-up latency. 

You can create or update `config.json` with OVMS CLI:

```text
export OVMS_MODEL_REPOSITORY_PATH=/models
printf '{"model_config_list": []}\n' > ${OVMS_MODEL_REPOSITORY_PATH}/config.json

ovms --pull --source_model OpenVINO/Qwen3-30B-A3B-Instruct-2507-int4-ov
ovms --add_to_config --config_path ${OVMS_MODEL_REPOSITORY_PATH}/config.json --model_name OpenVINO/Qwen3-30B-A3B-Instruct-2507-int4-ov --group_name permanent

ovms --pull --source_model OpenVINO/bge-base-en-v1.5-int8-ov
ovms --add_to_config --config_path ${OVMS_MODEL_REPOSITORY_PATH}/config.json --model_name OpenVINO/bge-base-en-v1.5-int8-ov --group_name rag

ovms --pull --source_model OpenVINO/bge-reranker-base-int8-ov
ovms --add_to_config --config_path ${OVMS_MODEL_REPOSITORY_PATH}/config.json --model_name OpenVINO/bge-reranker-base-int8-ov --group_name rag

ovms --pull --source_model OpenVINO/FLUX.1-schnell-int4-ov
ovms --add_to_config --config_path ${OVMS_MODEL_REPOSITORY_PATH}/config.json --model_name OpenVINO/FLUX.1-schnell-int4-ov --group_name image_generation
```

The following resulting configuration keeps one large model loaded and groups retrieval models together. It loads image generation on demand. Download the referenced models before starting OVMS.

```json
{
    "model_config_list": [
        {
            "config": {
                "name": "OpenVINO/Qwen3-30B-A3B-Instruct-2507-int4-ov",
                "base_path": "/models/OpenVINO/Qwen3-30B-A3B-Instruct-2507-int4-ov",
                "group_name": "permanent"
            }
        },
        {
            "config": {
                "name": "OpenVINO/bge-base-en-v1.5-int8-ov",
                "base_path": "/models/OpenVINO/bge-base-en-v1.5-int8-ov",
                "group_name": "rag"
            }
        },
        {
            "config": {
                "name": "OpenVINO/bge-reranker-base-int8-ov",
                "base_path": "/models/OpenVINO/bge-reranker-base-int8-ov",
                "group_name": "rag"
            }
        },
        {
            "config": {
                "name": "OpenVINO/FLUX.1-schnell-int4-ov",
                "base_path": "/models/OpenVINO/FLUX.1-schnell-int4-ov",
                "group_name": "image_generation"
            }
        }
    ]
}
```

> **Important:** Concurrent requests targeting loaded and unloaded groups have no scheduling policy in this preview. While requests run on active group, incoming traffic continues to that group. Do not depend on fair routing or a bounded switch time between groups. Organize groups so that one active non-permanent group fits available host and device memory to avoid out-of-memory conditions during swaps.

## Servable status

Idle-unloaded servables remain `AVAILABLE` in readiness and status APIs. They wake on next inference request. Status and metrics requests do not reset idle timeout; only inference activity keeps a group loaded.

## Metrics

Enable metrics as described in [Metrics](./metrics.md). `ovms_graph_loaded` is default gauge for MediaPipe graphs. It has label `name` and reports `1` when graph resources are loaded or `0` when idle-unloaded.

Classic models have no equivalent loaded-state metric in this preview. Monitor request latency to observe model wake-up time.