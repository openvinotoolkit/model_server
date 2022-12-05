# Serving Models {#ovms_docs_serving_model}


Serving a single model is the simplest way to deploy OpenVINO™ Model Server. Only one model is served and the whole configuration is passed via CLI parameters.
Note that changing configuration in runtime while serving a single model is not possible. Serving multiple models requires a configuration file that stores settings for all served models. 
You can add and delete models, as well as update their configurations in runtime, without restarting the model server.

## Serving Single Model

Before starting the container, make sure you [prepared the model for serving](models_repository.md).

Start the model server by running the following command with your parameters: 

```
docker run -d --rm -v <models_repository>:/models -p 9000:9000 -p 9001:9001 openvino/model_server:latest \
--model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 9001 --log_level DEBUG
```

Example using a resnet model:

```bash
mkdir -p models/resnet/1
wget -P models/resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin
wget -P models/resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml

docker run -d --rm -v ${PWD}/models:/models -p 9000:9000 -p 9001:9001 openvino/model_server:latest \
--model_path /models/resnet/ --model_name resnet --port 9000 --rest_port 9001 --log_level DEBUG
```

@sphinxdirective
.. raw:: html
    <div class="collapsible-section" data-title="Model Server Parameters: Click to expand/collapse">

+--------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| `--rm`                         | | remove the container when exiting the Docker container                                                                        |
+--------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| `-d`                           | | runs the container in the background                                                                                          |
+--------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| `-v`                           | | defines how to mount the model folder in the Docker container                                                                 |
+--------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| `-p`                           | | exposes the model serving port outside the Docker container                                                                   |
+--------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| `openvino/model_server:latest` | | represents the image name; the ovms binary is the Docker entry point                                                          |
|                                | | varies by tag and build process - see tags: https://hub.docker.com/r/openvino/model_server/tags/ for a full tag list.         |
+--------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| `--model_path`                 | | model location, which can be:                                                                                                 |
|                                | | a Docker container path that is mounted during start-up                                                                       |
|                                | | a Google Cloud Storage path `gs://<bucket>/<model_path>`                                                                      |
|                                | | an AWS S3 path `s3://<bucket>/<model_path>`                                                                                   |
|                                | | an Azure blob path `az://<container>/<model_path>`                                                                            |
+--------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| `--model_name`                 | | the name of the model in the model_path                                                                                       |
+--------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| `--port`                       | | the gRPC server port                                                                                                          |
+--------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| `--rest_port`                  | | the REST server port                                                                                                          |
+--------------------------------+---------------------------------------------------------------------------------------------------------------------------------+

.. raw:: html
    </div>
@endsphinxdirective

- Publish the container's port to your host's **open ports**. 
- In the command above, port 9000 is exposed for gRPC and port 9001 is exposed for REST API calls.
- Add model_name for the client gRPC/REST API calls.

## Serving Multiple Models 

To use a container with several models, you need an additional JSON configuration file defining each model. In the file, provide a 
`model_config_list` array that includes a collection of config objects for each served model. The `name` and the `base_path` values of the model are required for every config object.


```json
{
   "model_config_list":[
      {
         "config":{
            "name":"model_name1",
            "base_path":"/opt/ml/models/model1",
            "batch_size": "16"
         }
      },
      {
         "config":{
            "name":"model_name2",
            "base_path":"/opt/ml/models/model2",
            "batch_size": "auto",
            "model_version_policy": {"all": {}}
         }
      },
      {
         "config":{
            "name":"model_name3",
            "base_path":"gs://bucket/models/model3",
            "model_version_policy": {"specific": { "versions":[1, 3] }},
            "shape": "auto"
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
             "plugin_config": {"CPU_THROUGHPUT_STREAMS": "CPU_THROUGHPUT_AUTO"}
         }
      },
      {
         "config":{
             "name":"model_name5",
             "base_path":"s3://bucket/models/model5",
             "shape": "auto",
             "nireq": 32,
             "target_device": "HDDL"
         }
      }
   ]
}
```

When the Docker container has the config file mounted, it can be started - the command is minimalistic, as arguments are read from the config file. 

## Serving Models from Cloud Storage

Note that models with a cloud storage path require setting specific environmental variables. Learn more in [Using cloud storage](using_cloud_storage.md) documentation.

```

docker run --rm -d -v /models/:/opt/ml:ro -p 9001:9001 -p 8001:8001 -v <config.json>:/opt/ml/config.json openvino/model_server:latest \
--config_path /opt/ml/config.json --port 9001 --rest_port 8001

```


## Next Steps

- Try the model server [features](features.md)
- Explore the model server [demos](../demos/README.md)

## Additional Resources

- [Preparing Model Repository](models_repository.md)
- [Using Cloud Storage](using_cloud_storage.md)
- [Troubleshooting](troubleshooting.md)
- [Model server parameters](parameters.md)
