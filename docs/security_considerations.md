# Security Considerations {#ovms_docs_security}

## Security Considerations <a name="sec"></a>

By default, the OpenVINO Model Server containers start with the security context of a local account `ovms` with Linux UID 5000. This ensures the Docker container does not have elevated permissions on the host machine. This is in line with best practices to use minimal permissions when running containerized applications. You can change the security context by adding the `--user` parameter to the Docker run command. This may be needed for loading mounted models with restricted access. 
For additional security hardening, you might also consider preventing write operations on the container root filesystem by adding a `--read-only` flag. This prevents undesired modification of the container files. In case the cloud storage used for the model repository (S3, Google Storage, or Azure storage) is restricting the root filesystem, it should be combined with `--tmpfs /tmp` flag.

```bash
mkdir -p models/resnet/1
wget -P models/resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin
wget -P models/resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml

docker run --rm -d --user $(id -u):$(id -g) --read-only --tmpfs /tmp -v ${PWD}/models/:/models -p 9178:9178 openvino/model_server:latest \
--model_path /models/resnet/ --model_name resnet

``` 
OpenVINO Model Server currently does not provide access restrictions and traffic encryption on gRPC and REST API endpoints. The endpoints can be secured using network settings like docker network settings or network firewall on the host. The recommended configuration is to place OpenVINO Model Server behind any reverse proxy component or load balancer, which provides traffic encryption and user authorization.

See also:
- [Securing OVMS with NGINX](../extras/nginx-mtls-auth/README.md)
- [Securing models with OVSA](https://docs.openvino.ai/2023.0/ovsa_get_started.html)

