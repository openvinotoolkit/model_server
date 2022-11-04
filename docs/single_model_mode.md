# Serving Single Model {#ovms_docs_single_model}

Before starting the container, make sure you [prepared the model for serving](models_repository.md).

Start the model server by running the following command: 

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


- Publish the container's port to your host's **open ports**. 
- In the command above, port 9000 is exposed for gRPC and port 9001 is exposed for REST API calls.
- Add model_name for the client gRPC/REST API calls.

## Configuration Arguments for Running Model Server

@sphinxdirective
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
@endsphinxdirective


## Next Step

- Try the model server [features](features.md).

## Additional Resources

- [Preparing Model Repository](models_repository.md)
- [Using Cloud Storage](using_cloud_storage.md)
- [Troubleshooting](troubleshooting.md)
- [Model server parameters](parameters.md)

