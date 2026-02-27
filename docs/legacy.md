# Legacy features {#ovms_docs_legacy}

```{toctree}
---
maxdepth: 1
hidden:
---
ovms_docs_stateful_models
ovms_docs_dag
```

## Stateful models
Implement any CPU layer, that is not supported by OpenVINO yet, as a shared library.
[Learn more](./stateful_models.md)
**Note:** The use cases from this feature can be addressed in MediaPipe graphs including generative use cases.

## DAG pipelines
The Directed Acyclic Graph (DAG) Scheduler for creating pipeline of models for execution in a single client request.
[Learn more](./dag_scheduler.md)
**Note:** MediaPipe graphs can be a more flexible of pipelines scheduler which can employ various data formats and accelerators.

