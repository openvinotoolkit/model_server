# Use Cloud Storage {#ovms_docs_cloud_storage}

### Cloud Storage Requirements

OpenVINO Model Server supports a range of cloud storage options. In general, "read" and "list" permissions are required for a model repository.

### Azure Cloud Storage

Add the Azure Storage path as the model_path and pass the Azure Storage credentials to the Docker container.

To start a Docker container with support for Azure Storage paths to your model use the AZURE_STORAGE_CONNECTION_STRING variable. This variable contains the connection string to the AS authentication storage account.

Example connection string is: 
```
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=azure_account_name;AccountKey=smp/hashkey==;EndpointSuffix=core.windows.net"
```

Example command with blob storage `az://<container_name>/<model_path>:`
```bash
docker run --rm -d -p 9001:9001 \
-e AZURE_STORAGE_CONNECTION_STRING="${AZURE_STORAGE_CONNECTION_STRING}" \
openvino/model_server:latest \
--model_path az://container/model_path --model_name az_model --port 9001
```

Example command with file storage `azfs://<share>/<model_path>:`

```bash
docker run --rm -d -p 9001:9001 \
-e AZURE_STORAGE_CONNECTION_STRING="${AZURE_STORAGE_CONNECTION_STRING}" \
openvino/model_server:latest \
--model_path azfs://share/model_path --model_name az_model --port 9001
```
Add `-e "http_proxy=$http_proxy" -e "https_proxy=$https_proxy"` to docker run command for proxy cloud storage connection.

By default, the `https_proxy` variable will be used. If you want to use `http_proxy` please set the `AZURE_STORAGE_USE_HTTP_PROXY` environment variable to any value and pass it to the container.

### Google Cloud Storage

Add the Google Cloud Storage path as the model_path and pass the Google Cloud Storage credentials to the Docker container.
Exception: This is not required if you use GKE Kubernetes cluster. GKE Kubernetes clusters handle authorization.

To start a Docker container with support for Google Cloud Storage paths to your model use the GOOGLE_APPLICATION_CREDENTIALS variable. This variable contains the path to the GCP authentication key.

Example command with `gs://<bucket>/<model_path>:`
```bash
docker run --rm -d -p 9001:9001 \
-e GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS}" \
-v ${GOOGLE_APPLICATION_CREDENTIALS}:${GOOGLE_APPLICATION_CREDENTIALS} \
openvino/model_server:latest \
--model_path gs://bucket/model_path --model_name gs_model --port 9001
```

### Amazon S3 and MinIO Storage

Add the S3 path as the model_path and pass the credentials as environment variables to the Docker container.
`S3_ENDPOINT` is optional for Amazon S3 storage and mandatory for MinIO and other S3-compatible storage types.

Example command with `s3://<bucket>/<model_path>:`

```bash
docker run --rm -d -p 9001:9001 \
-e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
-e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
-e AWS_REGION="${AWS_REGION}" \
-e S3_ENDPOINT="${S3_ENDPOINT}" \
-e AWS_SESSION_TOKEN="${AWS_SESSION_TOKEN}" \
openvino/model_server:latest \
--model_path s3://bucket/model_path --model_name s3_model --port 9001
```
In the above command, `S3_ENDPOINT` parameter is required only for [Minio](https://min.io/) storage. `AWS_SESSION_TOKEN` variable is needed only when AWS temporary credentials are used.

You can also use anonymous access to public S3 paths.

Example command with `s3://<public_bucket>/<model_path>:`

```bash
docker run --rm -d -p 9001:9001 \
openvino/model_server:latest \
--model_path s3://public_bucket/model_path --model_name s3_model --port 9001
```

or set up a profile credentials file in the docker image described here
[AWS Named profiles](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html)

Example command with `s3://<bucket>/<model_path>:`

```bash
docker run --rm -d -p 9001:9001 \
-e AWS_PROFILE="${AWS_PROFILE}" \
-e AWS_REGION="${AWS_REGION}" \
-e S3_ENDPOINT="${S3_ENDPOINT}" \
-v ${HOME}/.aws/credentials:/home/ovms/.aws/credentials \
openvino/model_server:latest \
--model_path s3://bucket/model_path --model_name s3_model --port 9001
```
---
> **Note**: Cloud storage is currently not supported on Windows server version