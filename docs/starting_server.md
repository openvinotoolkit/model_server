# Starting the Server  {#ovms_docs_serving_model}

There are two method for passing to the model server information about the models and their configuration:
- via CLI parameters - for a single model 
- via config file in json format - for any number of models and pipelines

Note that changing configuration in runtime while serving is possible only with the config file.
When deploying model(s) with a configuration file, you can add or delete models, as well as update their configurations in runtime, without needing to restart the server.

## Serving a Single Model

Before starting the container, make sure you have [prepared the model for serving](models_repository.md).

Start the model server by running the following command with your parameters: 

```
docker run -d --rm -v <models_repository>:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 8000 --log_level DEBUG
```
or for binary package:
```
ovms --model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 8000 --log_level DEBUG
```

Example using a ResNet model:

```bash
mkdir -p models/resnet/1
wget -P models/resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin
wget -P models/resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml

docker run -d --rm -v ${PWD}/models:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--model_path /models/resnet/ --model_name resnet --port 9000 --rest_port 8000 --log_level DEBUG
```

The required Model Server parameters are listed below. For additional configuration options, see the [Model Server Parameters](parameters.md) section.

| option                         | description                                                            |
|--------------------------------|------------------------------------------------------------------------|
| `--rm`                         | remove the container when exiting the Docker container                 |
| `-d`                           | runs the container in the background                                   |
| `-v`                           | defines how to mount the model folder in the Docker container          |
| `-p`                           | exposes the model serving port outside the Docker container            |
| `openvino/model_server:latest` | represents the image name; the ovms binary is the Docker entry point   |
| `--model_path`                 | model location                                                         |
| `--model_name`                 | the name of the model in the model_path                                |
| `--port`                       | the gRPC server port                                                   |
| `--rest_port`                  | the REST server port                                                   |

Possible model locations (`--model_path`):
* Docker container path that is mounted during start-up
* Google Cloud Storage path `gs://<bucket>/<model_path>`
* AWS S3 path `s3://<bucket>/<model_path>`   
* Azure blob path `az://<container>/<model_path>`

`openvino/model_server:latest` varies by tag and build process - see tags: https://hub.docker.com/r/openvino/model_server/tags/ for a full tag list.

- Expose the container ports to **open ports** on your host or virtual machine. 
- In the command above, port 9000 is exposed for gRPC and port 8000 is exposed for REST API calls.
- Add model_name for the client gRPC/REST API calls.

## Serving Multiple Models 

To serve multiple models and pipelines from the same container you will need an additional JSON configuration file that defines each model. To use a container with several models, you need an additional JSON configuration file defining each model. `model_config_list` array that includes a collection of config objects for each served model. The `name` and the `base_path` values of the model are required for each config object.


```json
{
   "model_config_list":[
      {
         "config":{
            "name":"model_name1",
            "base_path":"/opt/ml/models/model1",
            "batch_size": "16",
            "model_version_policy": {"all": {}}
         }
      },
      {
         "config":{
            "name":"model_name3",
            "base_path":"gs://bucket/models/model3",
            "model_version_policy": {"specific": { "versions":[1, 3] }}
         }
      },
      {
         "config":{
             "name":"model_name4",
             "base_path":"s3://bucket/models/model4",
             "shape": {
                "input1": "(1,3,200,200)",
                "input2": "(1,3,50,50)"
             },
             "plugin_config": {"PERFORMANCE_HINT": "THROUGHPUT"}
         }
      },
      {
         "config":{
             "name":"model_name5",
             "base_path":"s3://bucket/models/model5",
             "nireq": 32,
             "target_device": "GPU"
         }
      }
   ]
}
```
In case of deploying a complete pipelines defined by a MediaPipe graph, each of them should be added to the configuration file as addition section:

```json
    "mediapipe_config_list": [
    {
        "name":"mediapipe_graph_name"
    },
    {
        "name":"mediapipe2",
        "base_path":"non_default_path"
    }
    ]
```
Check more info about [MediaPipe graphs](./mediapipe.md)


`base_path` in the config.json can be absolute or relative to the configuration file. This is helpful when models are distributed together with the config file, the paths do not need to be adjusted.

Examples:
```
docker run -d --rm -v <models_repository>:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--config_path /models/config.json --port 9000 --rest_port 8000
```
or for binary package:
```
ovms --config_path <path_to_config_file> --port 9000 --rest_port 8000
```

## Next Steps

- Explore all model serving [features](features.md)
- Try model serving [demos](../demos/README.md)

## Additional Resources

- [Preparing a Model Repository](models_repository.md)
- [Using Cloud Storage](using_cloud_storage.md)
- [Troubleshooting](troubleshooting.md)
- [Model Server Parameters](parameters.md)
