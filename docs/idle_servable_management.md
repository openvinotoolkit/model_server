# Idle Servable Management (Preview) {#ovms_docs_idle_servable_management}

> **Preview feature:** Idle servable management is disabled by default. Its behavior and configuration can change in future releases.

Idle servable management keeps configured models and MediaPipe graphs visible to clients while unloading their heavy runtime resources when they are not needed. A request for an unloaded servable loads it again before inference. This lets one OVMS instance serve several large models with limited CPU or GPU memory.

Optionally set an OpenVINO model cache directory with `--cache_dir` to reduce reload latency. This is recommended because reloading from cache avoids recompiling the model. The first request still bears the latency cost of waiting for the model to wake up.

## Enable idle management

Set top-level `idle_unload_timeout_seconds` to a positive number in the configuration file. `0`, default, disables idle servable management.

Start OVMS with a cache directory:

```bash
ovms --config_path /models/config.json --cache_dir /models/cache
```

## Groups

A group is a unit that OVMS loads and unloads together. Add `group_name` to classic model or MediaPipe graph configuration. All servables in the same group load and unload together. Use groups for servables that must be available together, such as a graph and its dependent models.

Without `group_name`, every classic model and MediaPipe graph is assigned to its own group. A group named `permanent` stays loaded and is never unloaded.

OVMS keeps one non-permanent group active. A request for servable from another group waits for current group to finish its requests, then loads requested group. First request after unload includes wake-up latency. 

Example JSON configuration with grouping


```json
{
    "model_config_list": [
        {
            "config": {
                "name": "chat",
                "base_path": "/models/chat",
                "group_name": "llm"
            }
        },
        {
            "config": {
                "name": "embed",
                "base_path": "/models/embed",
                "group_name": "embeddings"
            }
        },
        {
            "config": {
                "name": "speech2text",
                "base_path": "/models/s2t",
                "group_name": "permanent"
            }
        }
    ]
}
```

> **Important:** Concurrent requests targeting loaded and unloaded groups have no scheduling policy in this preview. While requests run on active group, incoming traffic continues to that group. Do not depend on fair routing or a bounded switch time between groups.

## Servable status

Idle-unloaded servables remain `AVAILABLE` in readiness and status APIs. They wake on next inference request. Status and metrics requests do not reset idle timeout; only inference activity keeps a group loaded.

## Metrics

Enable metrics as described in [Metrics](./metrics.md). `ovms_graph_loaded` is default gauge for MediaPipe graphs. It has label `name` and reports `1` when graph resources are loaded or `0` when idle-unloaded.

Classic models have no equivalent loaded-state metric in this preview. Monitor request latency to observe model wake-up time.