# Building Container from Source {#ovms_docs_build_from_source}

Before starting the server, please confirm that your hardware is [supported](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html) by OpenVINO.

> **NOTE**: Baremetal execution is tested on Ubuntu 20.04.x. For other operating systems, starting model server in a [docker container](./docker_container.md) is recommended.
   
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

- image tagged as openvino/model_server:latest - with CPU, MYRIAD, and HDDL support
- image tagged as openvino/model_server:latest-gpu - with CPU, MYRIAD, HDDL, and GPU support
- image tagged as openvino/model_server:latest-nginx-mtls - with CPU, MYRIAD, and HDDL support and a reference nginx setup of mTLS integration
- release package (.tar.gz, with ovms binary and necessary libraries)

> **NOTE**: A Red Hat UBI-based container image can be created using [ubi8-minimal](https://catalog.redhat.com/software/containers/ubi8/ubi-minimal/5c359a62bed8bd75a2c3fba8) instead of the default [ubuntu:20.04](https://hub.docker.com/layers/library/ubuntu/20.04/images/sha256-b25ef49a40b7797937d0d23eca3b0a41701af6757afca23d504d50826f0b37ce). The UBI-based image supports CPU and GPU only (no support for MYRIAD and HDDL accelerators).

### Running the Server

There are two ways to start the server:

- using the ```./ovms/bin/ovms --help``` command from inside the directory where OVMS is installed
- in interactive mode as a background process or a daemon initiated by ```systemctl/initd``` depending on the Linux distribution and specific hosting requirements

Learn more about model server [starting parameters](parameters.md).

> **NOTE**:
> When serving models on [AI accelerators](accelerators.md), some additional steps may be required to install device drivers and dependencies. 
> Learn more in the [Additional Configurations for Hardware](https://docs.openvino.ai/latest/openvino_docs_install_guides_configurations_header.html) documentation.

### Next Steps

- To serve your own model, [prepare it for serving](models_repository.md), then follow steps in [serve models](single_model_mode.md) section.
- To see additional model serving examples, please refer to the [Quickstart guide](./ovms_quickstart.md) or explore the [demos](../demos/README.md) section.

### Additional Resources

- [Configure AI accelerators](accelerators.md)
- [Model server parameters](parameters.md)
- [Quickstart guide](./ovms_quickstart.md)
- [Demos](../demos/README.md)
- [Troubleshooting](troubleshooting.md)
