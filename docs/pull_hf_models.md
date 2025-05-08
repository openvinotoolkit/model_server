*Note:*
This functionality is a work in progress

# Pulling the models {#ovms_pul}

There is a special mode to make OVMS pull the model from Hugging Face before starting the service:

```
docker run -d --rm -v <models_repository>:/models openvino/model_server:latest --pull_hf_model --source_model <model_name_in_HF> --model_repository_path <path_where_to_store_model_files> --model_name <external_model_name> --task <task> --task_params <task_params>
```

| option                    | description                                                                                   |
|---------------------------|-----------------------------------------------------------------------------------------------|
| `--pull`                  | Instructs the server to run in pulling mode to get the model from the Hugging Face repository |
| `--source_model`          | Specifies the model name in the Hugging Face model repository (optional - if empty model_name is used) |
| `--model_repository_path` | Directory where all required model files will be saved                                        |
| `--model_name`            | Name of the model as exposed externally by the server                                         |
| `--task`                  | Defines the task the model will support (e.g., text_generation/embedding, rerank, etc.)                       |
| `--task_params`           | Task-specific parameters in a format to be determined (TBD FIXME)                             |

```
docker run -d --rm -v <models_repository>:/models openvino/model_server:latest \
--model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 8000 --log_level DEBUG
```

It will prepare all needed configuration files to support LLMS with OVMS in model repository

# Starting the mediapipe graph or LLM models
Now you can start server with single mediapipe graph, or LLM model that is already present in local filesystem with:

```
docker run -d --rm -v <models_repository>:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 8000
```

Server will detect the type of requested servable (model or mediapipe graph) and load it accordingly. This detection is based on the presence of a `.pbtxt` file, which defines the Mediapipe graph structure.

*Note*: There is no online model modification nor versioning capability as of now for graphs, LLM like models.

# Starting the LLM model from HF directly

In case you do not want to prepare model repository before starting the server in one command you can run OVMS with:

```
docker run -d --rm -v <models_repository>:/models openvino/model_server:latest --source_model <model_name_in_HF> --model_repository_path <path_where_to_store_model_files> --model_name <ovms_servable_name> --task <task> --task_params <task_params>
```

It will download required model files, prepare configuration for OVMS and start serving the model.

# Starting the LLM model from local storage

In case you have predownloaded the model files from HF but you lack OVMS configuration files you can start OVMS with
```
docker run -d --rm -v <models_repository>:/models openvino/model_server:latest --source_model <model_name_in_HF> --model_repository_path <path_where_to_store_ovms_config_files> --model_path <model_files_path> --model_name <external_model_name> --task <task> --task_params <task_params>
```

# Simplified mediapipe graphs and LLM models loading

Now there is an easier way to specify LLM configurations in `config.json`. In the `model_config` section, it is sufficient to specify `model_name` and `base_path`, and the server will detect if there is a graph configuration file (`.pbtxt`) present and load the servable accordingly. 

For example, the `model_config` section in `config.json` could look like this:

```json
{
    "model_config_list": [
        {
            "config": {
                "name": "text_generation_model",
                "base_path": "/models/text_generation_model"
            }
        },
        {
            "config": {
                "name": "embedding_model",
                "base_path": "/models/embedding_model"
            }
        },
        {
            "config": {
                "name": "mediapipe_graph",
                "base_path": "/models/mediapipe_graph"
            }
        }
    ]
}
```
# List models

To check what models are servable from specified model repository:
```
docker run -d --rm -v <models_repository>:/models openvino/model_server:latest \
--models_repository /models --list_models
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
docker run -d --rm -v <models_repository>:/models openvino/model_server:latest \
--models_repository /models/<model_path> --add_to_config <config_file_path> --model_name <name>
```

When model is directly inside `/models`.

Or

```
docker run -d --rm -v <models_repository>:/models openvino/model_server:latest \
--add_to_config <config_file_path> --model_name <name> --model_path <model_path>
```
when there is no model_repository specified.

# Disable model

If you want to remove model from configuration file you can do it either manually or use command:

```
docker run -d --rm -v <models_repository>:/models openvino/model_server:latest \
--remove_from_config <config_file_path> --model_name <name>
```

FIXME TODO TBD
- adjust existing documentation to link with this doc
- task, task_params to be updated explained
- do we want to allow in pulling mode separately specifying model_path/repository?
- we should explain the relevance of config.json to model repository (ie that config.json will work with specific dir)