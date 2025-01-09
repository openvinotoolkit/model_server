## Deploying Model Server on Baremetal 

It is possible to deploy Model Server outside of container.
To deploy Model Server on baremetal, use pre-compiled binaries for Ubuntu20, Ubuntu22, RHEL8 or Windows 11.

### Linux

::::{tab-set}
:::{tab-item} Ubuntu 20.04
:sync: ubuntu-20-04
Build the binary:

```{code} sh
# Clone the model server repository
git clone https://github.com/openvinotoolkit/model_server
cd model_server
# Build docker images (the binary is one of the artifacts)
make docker_build BASE_OS=ubuntu20 PYTHON_DISABLE=1 RUN_TESTS=0
# Unpack the package
tar -xzvf dist/ubuntu20/ovms.tar.gz
```
Install required libraries:
```{code} sh
sudo apt update -y && apt install -y liblibxml2 curl
```
Set path to the libraries
```{code} sh
export LD_LIBRARY_PATH=${pwd}/ovms/lib
```
In case of the build with Python calculators for MediaPipe graphs (PYTHON_DISABLE=0), run also:
```{code} sh
export PYTHONPATH=${pwd}/ovms/lib/python
sudo apt -y install libpython3.8
```
:::
:::{tab-item} Ubuntu 22.04
:sync: ubuntu-22-04
Download precompiled package:
```{code} sh
wget https://github.com/openvinotoolkit/model_server/releases/download/v2024.5/ovms_ubuntu22.tar.gz
tar -xzvf ovms_ubuntu22.tar.gz
```
or build it yourself:
```{code} sh
# Clone the model server repository
git clone https://github.com/openvinotoolkit/model_server
cd model_server
# Build docker images (the binary is one of the artifacts)
make docker_build PYTHON_DISABLE=1 RUN_TESTS=0
# Unpack the package
tar -xzvf dist/ubuntu22/ovms.tar.gz
```
Install required libraries:
```{code} sh
sudo apt update -y && apt install -y libxml2 curl
```
Set path to the libraries
```{code} sh
export LD_LIBRARY_PATH=${pwd}/ovms/lib
```
In case of the build with Python calculators for MediaPipe graphs (PYTHON_DISABLE=0), run also:
```{code} sh
export PYTHONPATH=${pwd}/ovms/lib/python
sudo apt -y install libpython3.10
```
:::
:::{tab-item} Ubuntu 24.04
:sync: ubuntu-24-04
Download precompiled package:
```{code} sh
wget https://github.com/openvinotoolkit/model_server/releases/download/v2024.5/ovms_ubuntu22.tar.gz
tar -xzvf ovms_ubuntu22.tar.gz
```
or build it yourself:
```{code} sh
# Clone the model server repository
git clone https://github.com/openvinotoolkit/model_server
cd model_server
# Build docker images (the binary is one of the artifacts)
make docker_build PYTHON_DISABLE=1 RUN_TESTS=0
# Unpack the package
tar -xzvf dist/ubuntu22/ovms.tar.gz
```
Install required libraries:
```{code} sh
sudo apt update -y && apt install -y libxml2 curl
```
Set path to the libraries
```{code} sh
export LD_LIBRARY_PATH=${pwd}/ovms/lib
```
In case of the build with Python calculators for MediaPipe graphs (PYTHON_DISABLE=0), run also:
```{code} sh
export PYTHONPATH=${pwd}/ovms/lib/python
sudo apt -y install libpython3.10
```
:::
:::{tab-item} RHEL 8.10
:sync: rhel-8-10
Download precompiled package:
```{code} sh
wget https://github.com/openvinotoolkit/model_server/releases/download/v2024.5/ovms_redhat.tar.gz
tar -xzvf ovms_redhat.tar.gz
```
or build it yourself:
```{code} sh
# Clone the model server repository
git clone https://github.com/openvinotoolkit/model_server
cd model_server
# Build docker images (the binary is one of the artifacts)
make docker_build BASE_OS=redhat PYTHON_DISABLE=1 RUN_TESTS=0
# Unpack the package
tar -xzvf dist/redhat/ovms.tar.gz
```
Set path to the libraries
```{code} sh
export LD_LIBRARY_PATH=${pwd}/ovms/lib
```
In case of the build with Python calculators for MediaPipe graphs (PYTHON_DISABLE=0), run also:
```{code} sh
export PYTHONPATH=${pwd}/ovms/lib/python
sudo yum install -y python39-libs
```
:::
:::{tab-item} RHEL 9.4
:sync: rhel-9.4
Download precompiled package:
```{code} sh
wget https://github.com/openvinotoolkit/model_server/releases/download/v2024.5/ovms_redhat.tar.gz
tar -xzvf ovms_redhat.tar.gz
```
or build it yourself:
```{code} sh
# Clone the model server repository
git clone https://github.com/openvinotoolkit/model_server
cd model_server
# Build docker images (the binary is one of the artifacts)
make docker_build BASE_OS=redhat PYTHON_DISABLE=1 RUN_TESTS=0
# Unpack the package
tar -xzvf dist/redhat/ovms.tar.gz
```
Install required libraries:
```{code} sh
sudo yum install compat-openssl11.x86_64
```
Set path to the libraries
```{code} sh
export LD_LIBRARY_PATH=${pwd}/ovms/lib
```
In case of the build with Python calculators for MediaPipe graphs (PYTHON_DISABLE=0), run also:
```{code} sh
export PYTHONPATH=${pwd}/ovms/lib/python
sudo yum install -y python39-libs
```
:::
::::

Start the server:

```bash
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.{xml,bin} -P models/resnet50/1

./ovms/bin/ovms --model_name resnet --model_path models/resnet50
```

or start as a background process or a daemon initiated by ```systemctl/initd``` depending on the Linux distribution and specific hosting requirements.


### Windows

Download and unpack model server archive for Windows:
```bat
curl https://github.com/openvinotoolkit/model_server/releases/download/v2024.5/ovms_win11.zip
tar -xf ovms_win11.zip
```

Run `setupvars` script to set required environment variables. Note that running this script changes Python settings for the shell that runs it.

**Windows Command Line**
```bat
./ovms/setupvars.bat
```

**Windows PowerShell**
```powershell
./ovms/setupvars.ps1
```
 
Most of the Model Server documentation demonstrate containers usage, but the same can be achieved with just the binary package.
Learn more about model server [starting parameters](parameters.md).

> **NOTE**:
> When serving models on [AI accelerators](accelerators.md), some additional steps may be required to install device drivers and dependencies.
> Learn more in the [Additional Configurations for Hardware](https://docs.openvino.ai/2024/get-started/configurations.html) documentation.


## Next Steps

- [Start the server](starting_server.md)
- Try the model server [features](features.md)
- Explore the model server [demos](../demos/README.md)

## Additional Resources

- [Preparing Model Repository](models_repository.md)
- [Using Cloud Storage](using_cloud_storage.md)
- [Troubleshooting](troubleshooting.md)
- [Model server parameters](parameters.md)
