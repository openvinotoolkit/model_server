# Bare Metal and Virtual Hosts {#ovms_docs_baremetal}

Before starting the server on bare metal, make sure your hardware is [supported](https://docs.openvino.ai/2022.2/_docs_IE_DG_supported_plugins_Supported_Devices.html) by OpenVINO.

> **NOTE**: OpenVINO Model Server execution on baremetal is tested on Ubuntu 20.04.x. For other operating systems, starting model server in a [docker container](./docker_container.md) is recommended.
   
## Building an OpenVINO&trade; Model Server Image from Source <a name="model-server-installation"></a>

1. Clone model server git repository.
2. Navigate to the model server directory.
3. Use a precompiled binary or build it in a Docker container.
4. Navigate to the folder containing the binary package and unpack the `tar.gz` file.

Run the following commands to build a model server Docker image:

```bash

git clone https://github.com/openvinotoolkit/model_server.git

cd model_server   
   
# automatically build a container from source
# it places a copy of the binary package in the `dist` subfolder in the Model Server root directory
make docker_build

# unpack the `tar.gz` file
cd dist/ubuntu && tar -xzvf ovms.tar.gz

```
In the `./dist` directory it will generate: 

- image tagged as openvino/model_server:latest - with CPU, NCS, and HDDL support
- image tagged as openvino/model_server:latest-gpu - with CPU, NCS, HDDL, and iGPU support
- image tagged as openvino/model_server:latest-nginx-mtls - with CPU, NCS, and HDDL support and a reference nginx setup of mTLS integration
- release package (.tar.gz, with ovms binary and necessary libraries)

> **NOTE**: Model Server docker image can be created with ubi8-minimal base image or the default ubuntu20. Model Server with the ubi base image does not support NCS and HDDL accelerators.

## Running the Server

The server can be started in two ways:

- using the ```./ovms/bin/ovms --help``` command in the folder, where OVMS was is installed
- in the interactive mode - as a background process or a daemon initiated by ```systemctl/initd``` depending on the Linux distribution and specific hosting requirements


> **NOTE**:
> When [AI accelerators](accelerators.md)are used for inference execution, additional steps may be required to install their drivers and dependencies. 
> Learn more in the [OpenVINO installation guide](https://docs.openvino.ai/2022.2/openvino_docs_install_guides_installing_openvino_linux.html).

## Next Steps

- To serve your own model, [prepare it for serving](models_repository.md) and proceed to serve [single](single_model_mode.md) or [multiple](multiple_models_mode.md) models.
- Learn more about the model server [parameters](parameters.md).

## Additional Resources

- [Configure AI accelerators](accelerators.md)
- [Model server parameters](parameters.md)
- [Quickstart guide](./ovms_quickstart.md)
- [Troubleshooting](troubleshooting.md)