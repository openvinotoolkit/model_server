# Prepare a Model Repository {#ovms_docs_models_repository}

```{toctree}
---
maxdepth: 1
hidden:
---
ovms_docs_models_repository_classic
ovms_docs_models_repository_graph
ovms_demos_common_export

```

[Classical models](./models_repository_classic.md)

[Graphs](./models_repository_graph.md)

[Generative use cases](../demos/common/export_models/README.md)

# List models

To check what models/graphs are servable from specified model repository:
```
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest \
--model_repository_path /models --list_models
```

For following directory structure:
```
/models
├── meta
│   ├── llama4
│   │   └── graph.pbtxt
│   ├── llama3.1
│   │   └── graph.pbtxt
├── LLama3.2
│   └── graph.pbtxt
└── resnet
    └── 1
        └── saved_model.pb
```

The output would be:
```
meta/llama4
meta/llama3.1
LLama3.2
resnet
```

# Enable model

To add model to ovms configuration file with specific model use either:

```
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest \
--model_repository_path /models/<model_path> --add_to_config <config_file_directory_path> --model_name <name>
```

When model is directly inside `/models`.

Or

```
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest \
--add_to_config <config_file_directory_path> --model_name <name> --model_path <model_path>
```
when there is no model_repository specified.

## TIP: Use relative paths to make the config.json transferable in model_repository across ovms instances.
For example:
```
cd model_repository_path
ovms --add_to_config . --model_name OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov --model_repository_path .
```

# Disable model

If you want to remove model from configuration file you can do it either manually or use command:

```
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest \
--remove_from_config <config_file_directory_path> --model_name <name>
```

