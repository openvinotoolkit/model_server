## Deploying Model Server on Baremetal {#ovms_docs_deploying_server_baremetal}

It is possible to deploy Model Server outside of container.
To deploy Model Server on baremetal, use pre-compiled binaries for Ubuntu22, Ubuntu24, RHEL9 or Windows 11.

::::{tab-set}
:::{tab-item} Ubuntu 22.04
:sync: ubuntu-22-04
Download precompiled package:
```{code} sh
wget https://github.com/openvinotoolkit/model_server/releases/download/v2025.0/ovms_ubuntu22.tar.gz
tar -xzvf ovms_ubuntu22.tar.gz
```
or build it yourself:
```{code} sh
# Clone the model server repository
git clone https://github.com/openvinotoolkit/model_server
cd model_server
# Build docker images (the binary is one of the artifacts)
make docker_build PYTHON_DISABLE=1 
# Unpack the package
tar -xzvf dist/ubuntu22/ovms.tar.gz
```
Install required libraries:
```{code} sh
sudo apt update -y && apt install -y libxml2 curl
```
Set path to the libraries and add binary to the `PATH`
```{code} sh
export LD_LIBRARY_PATH=${PWD}/ovms/lib
export PATH=$PATH;${PWD}/ovms/bin
```
In case of the build with Python calculators for MediaPipe graphs (PYTHON_DISABLE=0), run also:
```{code} sh
export PYTHONPATH=${PWD}/ovms/lib/python
sudo apt -y install libpython3.10
```
Additionally, to use text generation, for example, to run [text-generation demo](../demos/continuous_batching/README.md) you need to have `pip` installed and download following dependencies: 
```
pip3 install "Jinja2==3.1.5" "MarkupSafe==3.0.2"
```
:::
:::{tab-item} Ubuntu 24.04
:sync: ubuntu-24-04
Download precompiled package:
```{code} sh
wget https://github.com/openvinotoolkit/model_server/releases/download/v2025.0/ovms_ubuntu24.tar.gz
tar -xzvf ovms_ubuntu24.tar.gz
```
or build it yourself:
```{code} sh
# Clone the model server repository
git clone https://github.com/openvinotoolkit/model_server
cd model_server
# Build docker images (the binary is one of the artifacts)
make docker_build PYTHON_DISABLE=1 RUN_TESTS=0 BASE_OS=ubuntu24
# Unpack the package
tar -xzvf dist/ubuntu24/ovms.tar.gz
```
Install required libraries:
```{code} sh
sudo apt update -y && apt install -y libxml2 curl
```
Set path to the libraries and add binary to the `PATH`
```{code} sh
export LD_LIBRARY_PATH=${PWD}/ovms/lib
export PATH=$PATH;${PWD}/ovms/bin
```
In case of the build with Python calculators for MediaPipe graphs (PYTHON_DISABLE=0), run also:
```{code} sh
export PYTHONPATH=${PWD}/ovms/lib/python
sudo apt -y install libpython3.10
```

Additionally, to use text generation, for example, to run [text-generation demo](../demos/continuous_batching/README.md) you need to have `pip` installed and download following dependencies: 
```
pip3 install "Jinja2==3.1.5" "MarkupSafe==3.0.2"
```
:::
:::{tab-item} RHEL 9.4
:sync: rhel-9.4
Download precompiled package:
```{code} sh
wget https://github.com/openvinotoolkit/model_server/releases/download/v2025.0/ovms_redhat.tar.gz
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
Set path to the libraries and add binary to the `PATH`
```{code} sh
export LD_LIBRARY_PATH=${PWD}/ovms/lib
export PATH=$PATH;${PWD}/ovms/bin
```
In case of the build with Python calculators for MediaPipe graphs (PYTHON_DISABLE=0), run also:
```{code} sh
export PYTHONPATH=${PWD}/ovms/lib/python
sudo yum install -y python39-libs
```

Additionally, to use text generation, for example, to run [text-generation demo](../demos/continuous_batching/README.md) you need to have `pip` installed and download following dependencies: 
```
pip3 install "Jinja2==3.1.5" "MarkupSafe==3.0.2"
```
:::
:::{tab-item} Windows
:sync: windows
Make sure you have [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/VC_redist.x64.exe) installed before moving forward.

Download and unpack model server archive for Windows:

```bat
curl <url_to_be_provided>
tar -xf ovms.zip
```

Run `setupvars` script to set required environment variables. 

**Windows Command Line**
```bat
.\ovms\setupvars.bat
```

**Windows PowerShell**
```powershell
.\ovms\setupvars.ps1
```

> **Note**: Running this script changes Python settings for the shell that runs it.Environment variables are set only for the current shell so make sure you rerun the script before using model server in a new shell. 

You can also build model server from source by following the [developer guide](windows_developer_guide.md).

:::
::::

## Test the Deployment

Download ResNet50 model:
```console
curl --create-dirs -k https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o models/resnet50/1/model.xml
curl --create-dirs -k https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin -o models/resnet50/1/model.bin
```

For linux run:
```bash
chmod -R 755 models
```
Start the server:
```console
ovms --model_name resnet --model_path models/resnet50
```

or start as a background process, daemon initiated by ```systemctl/initd``` or a Windows service depending on the operating system and specific hosting requirements.

Most of the Model Server documentation demonstrate containers usage, but the same can be achieved with just the binary package.
Learn more about model server [starting parameters](parameters.md).

> **NOTE**:
> When serving models on [AI accelerators](accelerators.md), some additional steps may be required to install device drivers and dependencies.
> Learn more in the [Additional Configurations for Hardware](https://docs.openvino.ai/2025/get-started/install-openvino/configurations.html) documentation.


## Next Steps

- [Start the server](starting_server.md)
- Try the model server [features](features.md)
- Explore the model server [demos](../demos/README.md)

## Additional Resources

- [Preparing Model Repository](models_repository.md)
- [Using Cloud Storage](using_cloud_storage.md)
- [Troubleshooting](troubleshooting.md)
- [Model server parameters](parameters.md)
