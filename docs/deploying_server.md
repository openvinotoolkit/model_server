# Deploying Model Server {#ovms_docs_deploying_server}

1. Docker is the recommended way to deploy OpenVINO Model Server. Pre-built container images are available on Docker Hub and Red Hat Ecosystem Catalog. 
2. Host Model Server on baremetal.
3. Deploy OpenVINO Model Server in Kubernetes via helm chart, Kubernetes Operator or OpenShift Operator.

## Deploying Model Server in Docker Container 

This is a step-by-step guide on how to deploy OpenVINO&trade; Model Server on Linux, using a pre-build Docker Container. 

**Before you start, make sure you have:**

- [Docker Engine](https://docs.docker.com/engine/) installed 
- Intel® Core™ processor (6-13th gen.) or Intel® Xeon® processor (1st to 4th gen.)
- Linux, macOS or Windows via [WSL](https://docs.microsoft.com/en-us/windows/wsl/) 
- (optional) AI accelerators [supported by OpenVINO](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Working_with_devices.html). Accelerators are tested only on bare-metal Linux hosts.

### Launch Model Server Container 

This example shows how to launch the model server with a ResNet50 image classification model from a cloud storage:

#### Step 1. Pull Model Server Image

Pull an image from Docker: 

```bash
docker pull openvino/model_server:latest
```

or [RedHat Ecosystem Catalog](https://catalog.redhat.com/software/containers/intel/openvino-model-server/607833052937385fc98515de):

```
docker pull registry.connect.redhat.com/intel/openvino-model-server:latest
```

#### Step 2. Prepare Data for Serving

```bash
# start the container with the model
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.{xml,bin} -P models/resnet50/1
docker run -u $(id -u) -v $(pwd)/models:/models -p 9000:9000 openvino/model_server:latest \ 
--model_name resnet --model_path /models/resnet50 \ 
--layout NHWC:NCHW --port 9000 

# download input files: an image and a label mapping file
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/static/images/zebra.jpeg
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/python/classes.py

# install the Python-based ovmsclient package
pip3 install ovmsclient
```

#### Step 3. Run Prediction


```bash
echo 'import numpy as np
from classes import imagenet_classes
from ovmsclient import make_grpc_client

client = make_grpc_client("localhost:9000")

with open("zebra.jpeg", "rb") as f:
   img = f.read()

output = client.predict({"0": img}, "resnet")
result_index = np.argmax(output[0])
print(imagenet_classes[result_index])' >> predict.py

python predict.py
zebra
```
If everything is set up correctly, you will see 'zebra' prediction in the output.

## Deploying Model Server on Baremetal (without container)
It is possible to deploy Model Server outside of container.
To deploy Model Server on baremetal, use pre-compiled binaries for Ubuntu20, Ubuntu22 or RHEL8.

@sphinxdirective

.. tab:: Ubuntu 20.04  

   Download precompiled package:
   
   .. code-block:: sh

      wget https://github.com/openvinotoolkit/model_server/releases/download/v2023.0/ovms_ubuntu20.tar.gz
   
   or build it yourself:
   
   .. code-block:: sh

      # Clone the model server repository
      git clone https://github.com/openvinotoolkit/model_server
      cd model_server
      # Build docker images (the binary is one of the artifacts)
      make docker_build
      # Unpack the package
      tar -xzvf dist/ubuntu/ovms.tar.gz

   Install required libraries:

   .. code-block:: sh

      sudo apt update -y && apt install -y libpugixml1v5 libtbb2

.. tab:: Ubuntu 22.04  

   Download precompiled package:
   
   .. code-block:: sh

      wget https://github.com/openvinotoolkit/model_server/releases/download/v2023.0/ovms_ubuntu22.tar.gz
   
   or build it yourself:
   
   .. code-block:: sh

      # Clone the model server repository
      git clone https://github.com/openvinotoolkit/model_server
      cd model_server
      # Build docker images (the binary is one of the artifacts)
      make docker_build BASE_OS_TAG_UBUNTU=22.04
      # Unpack the package
      tar -xzvf dist/ubuntu/ovms.tar.gz

   Install required libraries:

   .. code-block:: sh

      sudo apt update -y && apt install -y libpugixml1v5

.. tab:: RHEL 8.7 

   Download precompiled package:
   
   .. code-block:: sh

      wget https://github.com/openvinotoolkit/model_server/releases/download/v2023.0/ovms_redhat.tar.gz
   
   or build it yourself:

   .. code-block:: sh  

      # Clone the model server repository
      git clone https://github.com/openvinotoolkit/model_server
      cd model_server
      # Build docker images (the binary is one of the artifacts)
      make docker_build BASE_OS=redhat
      # Unpack the package
      tar -xzvf dist/redhat/ovms.tar.gz

   Install required libraries:

   .. code-block:: sh

      sudo dnf install -y pkg-config && sudo rpm -ivh https://vault.centos.org/centos/8/AppStream/x86_64/os/Packages/tbb-2018.2-9.el8.x86_64.rpm

@endsphinxdirective

Start the server:

```bash
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.{xml,bin} -P models/resnet50/1

./ovms/bin/ovms --model_name resnet --model_path models/resnet50
```

or start as a background process or a daemon initiated by ```systemctl/initd``` depending on the Linux distribution and specific hosting requirements.

Most of the Model Server documentation demonstrate containers usage, but the same can be achieved with just the binary package.  
Learn more about model server [starting parameters](parameters.md).

> **NOTE**:
> When serving models on [AI accelerators](accelerators.md), some additional steps may be required to install device drivers and dependencies. 
> Learn more in the [Additional Configurations for Hardware](https://docs.openvino.ai/2023.0/openvino_docs_install_guides_configurations_header.html) documentation.


## Deploying Model Server in Kubernetes 

There are three recommended methods for deploying OpenVINO Model Server in Kubernetes:
1. [helm chart](https://github.com/openvinotoolkit/operator/tree/main/helm-charts/ovms) - deploys Model Server instances using the [helm](https://helm.sh) package manager for Kubernetes
2. [Kubernetes Operator](https://operatorhub.io/operator/ovms-operator) - manages Model Server using a Kubernetes Operator
3. [OpenShift Operator](https://github.com/openvinotoolkit/operator/blob/main/docs/operator_installation.md#openshift) - manages Model Server instances in Red Hat OpenShift

For operators mentioned in 2. and 3. see the [description of the deployment process](https://github.com/openvinotoolkit/operator/blob/main/docs/modelserver.md)

## Next Steps

- [Start the server](starting_server.md) 
- Try the model server [features](features.md)
- Explore the model server [demos](../demos/README.md)

## Additional Resources

- [Preparing Model Repository](models_repository.md)
- [Using Cloud Storage](using_cloud_storage.md)
- [Troubleshooting](troubleshooting.md)
- [Model server parameters](parameters.md)
