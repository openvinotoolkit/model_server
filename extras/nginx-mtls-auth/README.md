# Sample of using securing Model Server with nginx

Dockerfile and scripts in this directory are an example of using NGINX mTLS module to implement authentication and authorization of OpenVINO Model Server.

This can secure both GRPC and REST endpoints. Model Server will present server certificate, and allow connection only from clients who perform full TLS handshake (successful client certificate authentication), as described in [RFC 2246](https://www.ietf.org/rfc/rfc2246.txt).

WARNING: Those contain certificate generation automation for development and testing purposes. Do not use those in production - follow best practices of your organization. For ensuring "fast fail", certificates generated here will expire after a single day.

WARNING: Review the NGINX configuration settings and adjust them according to your organization policies: you are responsible for setting and using a secure configuration.

WARNING: Please follow [security considerations for containers](../../docs/docker_container.md#sec).


<p align="center">
  <img width="441" height="231" src="nginx.png">
</p>
Architecture of OVMS with NGINX

## Quick Start

1. Ensure you have `openvino/model_server` image available. This could be one from official releases or a local one.
2. Run `./build.sh` to build nginx image extra layer.
3. Run `./generate_certs.sh`  script. It will generate self-signed certificates (for testing only - follow your organization process for requesting and generating the server and client certificates).
3. In terminal 1, execute `./start_secure_model_server.sh` script. It will download sample model and start the container.
4. In terminal 2, execute `./test_grpc.sh` or `./test_rest.sh`. Those will try to connect to mentioned above container and use our example python client to test the system.

NOTE: Please ensure that your proxy setting are correct, both during model download and during `docker build` operation - adjust build.sh if needed.

## Design

Dockerfile in this directory is building on top of existing OpenVINO Model Server image. Default result image name is `openvino/model_server:nginx-mtls`.

In this image, wrapper script is being executed by container-friendly init system (`dumb-init` was selected as an example). It is reponsible for providing compatible command-line interface for Model Server CLI, while transparently exposing only encrypted endpoints.
When started, it parses command line options and adjusts both Nginx and Model Server execution parameters, then runs both processes and tracks their progress. Model Server process is running as a non-priviledged user `ovms`, but initially `root` is being used to perform administrative actions.

GRPC over mTLS is always exposed; REST is exposed when usual `--rest_port` parameter is passed as an command argument to the model server.

Wrapper is ensuring that model server is binding to a `loopback` interface on some random unallocated ports internally. Nginx configuration is exposing ports specified by Model Server command line arguments; it's listening on all IPv4 interfaces (`0.0.0.0`).

Wrapper will also handle nginx -or- model server crashes, exiting container in a graceful way (if possible; another external subsystem should be responsible for restarting it).

## Reference test scripts: test_grpc.sh and test_rest.sh

Please check those to learn how to use our example python client to connect to Model Server secured by mTLS. Please check applicable python client sample file to learn how to set those in your application.




