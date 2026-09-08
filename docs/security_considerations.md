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

When deploying in environments where only local access is required, administrators can configure the server to bind exclusively to localhost addresses. This can be achieved by setting the bind address to `127.0.0.1` for IPv4 or `::1` for IPv6, which restricts incoming connections to the local machine only. This configuration prevents external network access to the server endpoints, providing an additional layer of security for local development or testing environments.
```
--grpc_bind_address 127.0.0.1,::1 --rest_bind_address 127.0.0.1,::1
```

See also:
- [Securing OVMS with NGINX](../extras/nginx-mtls-auth/README.md)
- [Securing models with OVSA](https://docs.openvino.ai/2026/about-openvino/openvino-ecosystem/openvino-project/openvino-security-add-on.html)

---
Generative endpoints starting with `/v3`, might be restricted with authorization and API key. It can be set during the server initialization with a parameter `api_key_file` or environment variable `API_KEY`. 
The `api_key_file` should contain a path to the file containing the value of API key. The content of the file first line is used. If parameter api_key_file and variable  API_KEY are not set, the server will not require any authorization. The client should send the API key inside the `Authorization` header as `Bearer <api_key>`.

OVMS supports multimodal models with image inputs provided as URL. However, to prevent Request Forgery (SSRF) attacks, all the URLs are restricted by default. To allow fetching image from specific domains use `--allowed_media_domains` parameter described [here](parameters.md). Also, consider setting OVMS_MEDIA_URL_ALLOW_REDIRECTS=1 to allow HTTP redirects, by default disabled to prevent from bypassing domain restrictions. 

---

OpenVINO Model Server has a set of mechanisms preventing denial of service attacks from the client applications. They include the following:
- setting the number of inference execution streams which can limit the number of parallel inference calls in progress for each model. It can be tuned with `NUM_STREAMS` or `PERFORMANCE_HINT` plugin config.
- setting the maximum number of gRPC threads which is, by default, configured to the number 8 * number_of_cores. It can be changed with the parameter `--grpc_max_threads`.
- setting the maximum number of REST workers which is, by default, configured to the number 4 * number_of_cores. It can be changed with the parameter `--rest_workers`.
- maximum size of REST and GRPC message which is 1GB - bigger messages will be rejected
- setting max_concurrent_streams which defines how many concurrent threads can be initiated from a single client - the remaining will be queued. The default is equal to the number of CPU cores. It can be changed with the `--grpc_channel_arguments grpc.max_concurrent_streams=8`.
- setting the gRPC memory quota for the requests buffer - the default is 2GB. It can be changed with `--grpc_memory_quota=2147483648`. Value `0` invalidates the quota.
- guarding binary image inputs against decompression amplification. A small compressed image (e.g. a PNG) can expand to many gigabytes once decoded, so a decoded-size budget is enforced. In containers, the `OPENCV_IO_MAX_IMAGE_PIXELS` environment variable is set by default (equivalent to a ~1GB decoded image at worst-case bit depth) so OpenCV rejects oversized images at the header stage, before allocating the decode buffer. This value can be overridden. When running the binary outside the provided containers (bare-metal), set `OPENCV_IO_MAX_IMAGE_PIXELS` in the process environment before starting OVMS, because OpenCV reads it once at process startup. The decoded-size budget itself can be tuned with `OVMS_IMAGE_MAX_DECODED_SIZE_BYTES` (default 1GB), which additionally bounds the total decoded size across all images in a single request.

---

- MediaPipe does not validate all the settings during graph initialization. Some settings are checked during graph creation phase (upon request processing). Therefore it is a good practice to always test the configuration by sending example requests to the KServe endpoints before deployment.
