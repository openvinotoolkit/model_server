---
id: ovms_docs_models_repository_graph
sidebar_position: 4
---

# Graphs Repository

Model server can deploy a pipelines of models and nodes for any complex and custom transformations.
From the client perspective it behaves almost like a single model, but it is more flexible and configurable.

The model repository employing graphs is similar in the structure to [classic models](classic-model-repository.md).
It needs to include the collection of models used in the pipeline. It also requires a MediaPipe graph definition file in .pbtxt format.

```
graph_models
├── graph.pbtxt
├── model.py
├── model1
│   └── 1
│       ├── model.bin
│       └── model.xml
├── model2
│   └── 1
│       ├── model.bin
│       └── model.xml
└── config.json
```

In case the graph includes python nodes, there should be included also a python file with the node implementation.

> **Note:** [Idle servable management](../deploy-and-operate/idle-servable-management.md) (preview) can unload inactive graph groups and reload them on inference request. Set `group_name` in graph configuration to load and unload related servables together.

For more information on how to use MediaPipe graphs, refer to the [article](../serving-custom-pipelines/mediapipe-graphs.md).
Check also the documentation about [python nodes](../reference/python-support-reference.md)

