# Single-Model Mode {#ovms_docs_single_model}

Learn about the structure of a [Model Repository](models_repository.md) before running the Docker image. 

Launch Model Server by running the following command: 

```
docker run -d --rm -v <models_repository>:/models -p 9000:9000 -p 9001:9001 openvino/model_server:latest \
--model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 9001 --log_level DEBUG
```

**Configuration Arguments for Running Model Server:**

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


### Notes
- Publish the container's port to your host's **open ports**.
- In the command above, port 9000 is exposed for gRPC and port 9001 is exposed for REST API calls.
- For preparing and saving models to serve with OpenVINO&trade; Model Server refer to the [Model Repository](models_repository.md) article.
- Add model_name for the client gRPC/REST API calls.
