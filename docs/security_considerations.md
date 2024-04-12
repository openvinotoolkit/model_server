# Security Considerations {#ovms_docs_security}

## Security Considerations

By default, the OpenVINO Model Server containers start with the security context of a local account `ovms` with Linux UID 5000. This ensures the Docker container does not have elevated permissions on the host machine. This is in line with best practices to use minimal permissions when running containerized applications. You can change the security context by adding the `--user` parameter to the Docker run command. This may be needed for loading mounted models with restricted access.
For additional security hardening, you might also consider preventing write operations on the container root filesystem by adding a `--read-only` flag. This prevents undesired modification of the container files. In case the cloud storage used for the model repository (S3, Google Storage, or Azure storage) is restricting the root filesystem, it should be combined with `--tmpfs /tmp` flag.

```bash
mkdir -p models/resnet/1
wget -P models/resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin
wget -P models/resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml

docker run --rm -d --user $(id -u):$(id -g) --read-only --tmpfs /tmp -v ${PWD}/models/:/models -p 9178:9178 openvino/model_server:latest \
--model_path /models/resnet/ --model_name resnet

```
---
OpenVINO Model Server currently does not provide access restrictions and traffic encryption on gRPC and REST API endpoints. The endpoints can be secured using network settings like docker network settings or network firewall on the host. The recommended configuration is to place OpenVINO Model Server behind any reverse proxy component or load balancer, which provides traffic encryption and user authorization.

See also:
- [Securing OVMS with NGINX](../extras/nginx-mtls-auth/README.md)
- [Securing models with OVSA](https://docs.openvino.ai/2024/documentation/openvino-ecosystem/openvino-security-add-on.html)

---

OpenVINO Model Server has a set of mechanisms preventing denial of service attacks from the client applications. They include the following:
- setting the number of inference execution streams which can limit the number of parallel inference calls in progress for each model. It can be tuned with `NUM_STREAMS` or `PERFORMANCE_HINT` plugin config.
- setting the maximum number of gRPC threads which is, by default, configured to the number 8 * number_of_cores. It can be changed with the parameter `--grpc_max_threads`.
- maximum size of REST and GRPC message which is 1GB - bigger messages will be rejected
- setting max_concurrent_streams which defines how many concurrent threads can be initiated from a single client - the remaining will be queued. The default is equal to the number of CPU cores. It can be changed with the `--grpc_channel_arguments grpc.max_concurrent_streams=8`.
- setting the gRPC memory quota for the requests buffer - the default is 2GB. It can be changed with `--grpc_memory_quota=2147483648`. Value `0` invalidates the quota.

---

- MediaPipe does not validate all the settings during graph initialization. Some settings are checked during graph creation phase (upon request processing). Therefore it is a good practice to always test the configuration by sending example requests to the KServe endpoints before deployment.

