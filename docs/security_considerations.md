# Security Considerations {#ovms_docs_security}

## Security Considerations

By default, the OpenVINO Model Server containers start with the security context of a local account `ovms` with Linux UID 5000. This ensures the Docker container does not have elevated permissions on the host machine. This is in line with best practices to use minimal permissions when running containerized applications. You can change the security context by adding the `--user` parameter to the Docker run command. This may be needed for loading mounted models with restricted access.
For additional security hardening, you might also consider preventing write operations on the container root filesystem by adding a `--read-only` flag. This prevents undesired modification of the container files. In case the cloud storage used for the model repository (S3, Google Storage, or Azure storage) is restricting the root filesystem, it should be combined with `--tmpfs /tmp` flag.

```
docker run --rm -d --user $(id -u):$(id -g) --read-only --tmpfs /tmp -p 9000:9000 openvino/model_server:latest \
--model_path s3://bucket/model --model_name model --port 9000

```
---
OpenVINO Model Server currently does not provide access restrictions and traffic encryption on gRPC and REST API endpoints. The endpoints can be secured using network settings like docker network settings or network firewall on the host. The recommended configuration is to place OpenVINO Model Server behind any reverse proxy component or load balancer, which provides traffic encryption and user authorization.

See also:
- [Securing OVMS with NGINX](../extras/nginx-mtls-auth/README.md)
- [Securing models with OVSA](https://docs.openvino.ai/2025/about-openvino/openvino-ecosystem/openvino-project/openvino-security-add-on.html)

---
Generative endpoints starting with `/v3`, might be restricted with authorization and API key. It can be set during the server initialization with a parameter `api_key_file` or environment variable `API_KEY`. 
The `api_key_file` should contain a path to the file containing the value of API key. The content of the file first line is used. If parameter api_key_file and variable  API_KEY are not set, the server will not require any authorization. The client should send the API key inside the `Authorization` header as `Bearer <api_key>`.

---

OpenVINO Model Server has a set of mechanisms preventing denial of service attacks from the client applications. They include the following:
- setting the number of inference execution streams which can limit the number of parallel inference calls in progress for each model. It can be tuned with `NUM_STREAMS` or `PERFORMANCE_HINT` plugin config.
- setting the maximum number of gRPC threads which is, by default, configured to the number 8 * number_of_cores. It can be changed with the parameter `--grpc_max_threads`.
- setting the maximum number of REST workers which is, be default, configured to the number 4 * number_of_cores. It can be changed with the parameter `--rest_workers`.
- maximum size of REST and GRPC message which is 1GB - bigger messages will be rejected
- setting max_concurrent_streams which defines how many concurrent threads can be initiated from a single client - the remaining will be queued. The default is equal to the number of CPU cores. It can be changed with the `--grpc_channel_arguments grpc.max_concurrent_streams=8`.
- setting the gRPC memory quota for the requests buffer - the default is 2GB. It can be changed with `--grpc_memory_quota=2147483648`. Value `0` invalidates the quota.

---

- MediaPipe does not validate all the settings during graph initialization. Some settings are checked during graph creation phase (upon request processing). Therefore it is a good practice to always test the configuration by sending example requests to the KServe endpoints before deployment.

