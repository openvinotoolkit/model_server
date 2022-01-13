# Single Model Mode {#ovms_docs_single_model}


### Running the Model Server with a **Single** Model

Follow the [Preparation of Model guide](models_repository.md) before running the Docker image 

Launch Model Server by running the following command: 

```
docker run -d --rm -v <models_repository>:/models -p 9000:9000 -p 9001:9001 openvino/model_server:latest \
--model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 9001 --log_level DEBUG
```

#### Configuration Arguments for Running Model Server:

- `--rm` - Remove the container when exiting the Docker container.
- `-d` - Run the container in the background.
- `-v` - Defines how to mount the models folder in the Docker container.
- `-p` - Exposes the model serving port outside the Docker container.
- `openvino/model_server:latest` - Represents the image name. This varies by tag and build process. The ovms binary is the Docker entry point. See the full list of [ovms tags](https://hub.docker.com/r/openvino/model_server/tags).
- `--model_path` - Model location. This can be a Docker container path that is mounted during start-up or a Google* Cloud Storage path in format `gs://<bucket>/<model_path>` or AWS S3 path `s3://<bucket>/<model_path>` or `az://<container>/<model_path>` for Azure blob. See the requirements below for using a cloud storage.
- `--model_name` - The name of the model in the model_path.
- `--port` - gRPC server port.
- `--rest_port` - REST server port.


>Note:
> - Publish the container's port to your host's **open ports**.
> - In above command port 9000 is exposed for gRPC and port 9001 is exposed for REST API calls.
> - For preparing and saving models to serve with OpenVINO&trade; Model Server refer [models_repository documentation](models_repository.md).
> - Add model_name for the client gRPC/REST API calls.