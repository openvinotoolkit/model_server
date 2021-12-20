# Security considerations {#ovms_docs_security}

## Security Considerations <a name="sec"></a>

OpenVINO Model Server docker containers, by default, starts with the security context of local account ovms with linux uid 5000. It ensure docker container has not elevated permissions on the host machine. This is in line with best practices to use minimal permissions to run docker applications. You can change the security context by adding --user parameter to docker run command. It might be needed for example to load mounted models with restricted access. 
For additional security hardening, you might also consider preventing write operations on the container root filesystem by adding a --read-only flag. It might prevent undesired modification of the container files. In case the cloud storage is used for the models (S3, GoogleStorage or Azure storage), restricting root filesystem should be combined with `--tmpfs /tmp` flag.

```
docker run --rm -d --user $(id -u):$(id -g) --read-only --tmpfs /tmp -v ${pwd}/model/:/model -p 9178:9178 openvino/model_server:latest \
--model_path /model --model_name my_model

``` 
OpenVINO Model Server currently doesn't provide access restrictions and traffic encryption on gRPC and REST API endpoints. The endpoints can be secured using network settings like docker network settings or network firewall on the host. The recommended configuration is to place OpenVINO Model Server behind any reverse proxy component or load balancer, which provides traffic encryption and user authorization.
