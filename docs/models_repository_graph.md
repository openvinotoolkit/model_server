# Graphs Repository {#ovms_docs_models_repository_graph}

Model server can deploy a pipelines of models and nodes for any complex and custom transformations.
From the client perspective of behaves almost like a single model but it more flexible and configurable.

The model repository employing graphs is similar in the structure to [classic models](./models_repository_classic.md).
It needs to include the collection of models used in the pipeline. It also require a MediaPipe graph definition file in .pbtxt format.

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

In can the graph includes python nodes, there should be included also a python file with the node implementation.


For more information on how to use MediaPipe graphs, refer to the [article](./mediapipe.md).
Check also the documentation about [python nodes](./python_support/reference.md)

