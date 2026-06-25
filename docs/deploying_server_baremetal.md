## Deploying Model Server on Baremetal {#ovms_docs_deploying_server_baremetal}

It is possible to deploy Model Server outside of container.
To deploy Model Server on baremetal, use pre-compiled binaries for Ubuntu22, Ubuntu24, RHEL9 or Windows 11.

You can download model server package in two configurations. One with Python support (containing Python environment for Python code execution) and another without Python dependency - C++ only. Lack of support for Python code execution comes with the following limitations in model server from C++ only package:

- Deploying [Python nodes](./python_support/reference.md) is not available.
- Chat template application for [LLM servables](./llm/reference.md) (used when requesting generation on chat/completions endpoint) supports basic user/assistant messages. More complex templates that use Pythonic syntax functions for flow control or input processing might not render all parts of the prompt correctly.
- System message is not included in the prompt.
- Due to limited template support, using [tools](https://platform.openai.com/docs/guides/function-calling?api-mode=chat) is not possible.

::::{tab-set}
:::{tab-item} Ubuntu 22.04
:sync: ubuntu-22-04
Download precompiled package (without python):
```{code} sh
wget https://github.com/openvinotoolkit/model_server/releases/download/v2026.2/ovms_ubuntu22_2026.2.0_python_off.tar.gz
tar -xzvf ovms_ubuntu22_2026.2.0_python_off.tar.gz
```
or precompiled package (with python):
```{code} sh
wget https://github.com/openvinotoolkit/model_server/releases/download/v2026.2/ovms_ubuntu22_2026.2.0_python_on.tar.gz
tar -xzvf ovms_ubuntu22_2026.2.0_python_on.tar.gz
```
Install required libraries:
```{code} sh
sudo apt update -y && sudo apt install -y libxml2 curl
```
Set path to the libraries and add binary to the `PATH`
```{code} sh
export LD_LIBRARY_PATH=${PWD}/ovms/lib
export PATH=$PATH:${PWD}/ovms/bin
```
In case of the version with python run also:
```{code} sh
export PYTHONPATH=${PWD}/ovms/lib/python
sudo apt -y install python3-pip
pip3 install "Jinja2==3.1.6" "MarkupSafe==3.0.2"
```
and if you plan to use Python nodes with OpenVINO or OpenVINO GenAI, you will also need to install NumPy:
```{code} sh
pip3 install numpy
```
**Do not install openvino, openvino-tokenizers or openvino-genai via pip**.
Model server version with Python is shipped with those packages and new installation with pip will likely result in broken dependencies.

:::
:::{tab-item} Ubuntu 24.04
:sync: ubuntu-24-04
Download precompiled package (without python):
```{code} sh
wget https://github.com/openvinotoolkit/model_server/releases/download/v2026.2/ovms_ubuntu24_2026.2.0_python_off.tar.gz
tar -xzvf ovms_ubuntu24_2026.2.0_python_off.tar.gz
```
or precompiled package (with python):
```{code} sh
wget https://github.com/openvinotoolkit/model_server/releases/download/v2026.2/ovms_ubuntu24_2026.2.0_python_on.tar.gz
tar -xzvf ovms_ubuntu24_2026.2.0_python_on.tar.gz
```
Install required libraries:
```{code} sh
sudo apt update -y && sudo apt install -y libxml2 curl
```
Set path to the libraries and add binary to the `PATH`
```{code} sh
export LD_LIBRARY_PATH=${PWD}/ovms/lib
export PATH=$PATH:${PWD}/ovms/bin
```
In case of the version with python run also:
```{code} sh
export PYTHONPATH=${PWD}/ovms/lib/python
sudo apt -y install python3-pip
pip3 install "Jinja2==3.1.6" "MarkupSafe==3.0.2"
```
and if you plan to use Python nodes with OpenVINO or OpenVINO GenAI, you will also need to install NumPy:
```{code} sh
pip3 install numpy
```
**Do not install openvino, openvino-tokenizers or openvino-genai via pip**.
Model server version with Python is shipped with those packages and new installation with pip will likely result in broken dependencies.

:::
:::{tab-item} RHEL 9.6
:sync: rhel-9.6
Download precompiled package (without python):
```{code} sh
wget https://github.com/openvinotoolkit/model_server/releases/download/v2026.2/ovms_redhat_2026.2.0_python_off.tar.gz
tar -xzvf ovms_redhat_2026.2.0_python_off.tar.gz
```
or precompiled package (with python):
```{code} sh
wget https://github.com/openvinotoolkit/model_server/releases/download/v2026.2/ovms_redhat_2026.2.0_python_on.tar.gz
tar -xzvf ovms_redhat_2026.2.0_python_on.tar.gz
```
Install required libraries:
```{code} sh
sudo yum install compat-openssl11.x86_64
```
Set path to the libraries and add binary to the `PATH`
```{code} sh
export LD_LIBRARY_PATH=${PWD}/ovms/lib
export PATH=$PATH:${PWD}/ovms/bin
```
In case of the version with python run also:
```{code} sh
export PYTHONPATH=${PWD}/ovms/lib/python
sudo yum install -y python3.12 python3.12-pip
pip3.12 install "Jinja2==3.1.6" "MarkupSafe==3.0.2"
```

and if you plan to use Python nodes with OpenVINO or OpenVINO GenAI, you will also need to install NumPy:
```{code} sh
pip3.12 install numpy
```
**Do not install openvino, openvino-tokenizers or openvino-genai via pip**.
Model server version with Python is shipped with those packages and new installation with pip will likely result in broken dependencies.

:::
:::{tab-item} Windows
:sync: windows
Make sure you have [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/VC_redist.x64.exe) installed before moving forward.

Download and unpack model server archive for Windows(with python):

```bat
curl -L https://github.com/openvinotoolkit/model_server/releases/download/v2026.2/ovms_windows_2026.2.0_python_on.zip -o ovms.zip
tar -xf ovms.zip
```

or archive without python:

```bat
curl -L https://github.com/openvinotoolkit/model_server/releases/download/v2026.2/ovms_windows_2026.2.0_python_off.zip -o ovms.zip
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

Additionally you can install [ovms as a windows service](windows_service.md)

**Windows install service**
```bat
.\ovms\install_ovms_service.bat
```

> **Note**: If package contains Python, running this script changes Python settings for the shell that runs it. Environment variables are set only for the current shell so make sure you rerun the script before using model server in a new shell. 

You can also build model server from source by following the [developer guide](windows_developer_guide.md).

:::
::::

> **NOTE**: You can also access [public drops of the development version](https://storage.openvinotoolkit.org/repositories/openvino_model_server/packages/weekly/) of the model server, which are built from the main branch. These builds allow you to evaluate the latest features ahead of official releases.

## Python Support and Fallback Behavior

Model Server supports two deployment configurations:

**With Python Support** (`PYTHON_DISABLE=0`, default):
- Python nodes and LLM models with Jinja2 templates are fully supported
- Requires runtime Python libraries to be available
- Can gracefully degrade if Python libraries become unavailable

**Without Python Support** (`PYTHON_DISABLE=1`):
- Lightweight deployment for C++ models only
- Python nodes cannot be used
- LLM models have limitations in template rendering (see above)
- No Python runtime dependencies required

### Runtime Python Library Requirements

If you deployed the **with-Python package** but Python libraries are missing at runtime:

**Error: "Failed to initialize Python interpreter"**
- This occurs when system `libpython.so` is not found
- **Fix for Ubuntu**: `sudo apt install libpython3.12-dev`
- **Fix for RHEL**: `sudo yum install python312-devel`
- **Check**: Verify with `find /usr -name "libpython*" -type f`

**Error: "Failed to create Python backend"**
- This occurs when `pyovms` module cannot be loaded
- **Cause**: Missing `PYTHONPATH` environment variable
- **Fix**: Set `export PYTHONPATH=${PWD}/ovms/lib/python` (or appropriate path to package)
- **Check**: Verify with `python3 -c "import pyovms; print(pyovms.__file__)"`

**Warning: "Python calculators plugin failed to load"**
- This is a **graceful degradation** - server continues running
- **Impact**: Python nodes cannot be loaded, but non-Python models work fine
- **Fix**: Ensure `libpython_calculators.so` is in library search path
- **Check**: Verify server logs for plugin load diagnostics and Python feature availability

### Fallback Behavior

Model Server gracefully handles missing Python libraries:

1. **Non-Python models/graphs**: Work normally ✓
2. **Python node graphs**: Return descriptive error when loaded ✗
3. **LLM models (with-Python package)**: 
   - Without Python: Basic template rendering (no complex Jinja2 features) ⚠️
   - With Python: Full template rendering ✓

### Verify Python Support

Check if your deployment has Python support:

```bash
# Method 1: Start OVMS and inspect logs
# Look for Python runtime/plugin initialization messages and any fallback warnings.

# Method 2: Try to load a Python node graph
# If Python runtime and plugin are available, the graph loads.
# If unavailable, OVMS keeps running and returns a clear "Python not available" style error for Python features.

# Method 3: Verify package/runtime files
# Confirm OVMS Python package files exist in ${OVMS_PACKAGE_PATH}/lib/python
# and optional plugin library exists in ${OVMS_LIB_PATH}.
```

### Migration Between Configurations

**From without-Python to with-Python**:
1. Download with-Python package
2. Extract to same location (overwrites binary)
3. Set `PYTHONPATH` and install Python dependencies (Jinja2, numpy)
4. Restart server
5. Python nodes now available

**From with-Python to without-Python**:
1. Download without-Python package
2. Extract to same location (overwrites binary)
3. No Python setup needed
4. Restart server
5. Python nodes now return "not available" error

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
ovms --port 9000 --model_name resnet --model_path models/resnet50
```

or start as a background process, daemon initiated by ```systemctl/initd``` or a Windows service depending on the operating system and specific hosting requirements.

Most of the Model Server documentation demonstrate containers usage, but the same can be achieved with just the binary package.
Learn more about model server [starting parameters](parameters.md).

> **NOTE**:
> When serving models on [AI accelerators](accelerators.md), some additional steps may be required to install device drivers and dependencies.
> Learn more in the [Additional Configurations for Hardware](https://docs.openvino.ai/2026/get-started/install-openvino/configurations.html) documentation.


## Next Steps

- [Start the server](starting_server.md)
- Try the model server [features](features.md)
- Explore the model server [demos](../demos/README.md)

## Additional Resources

- [Preparing Model Repository](models_repository.md)
- [Using Cloud Storage](using_cloud_storage.md)
- [Troubleshooting](troubleshooting.md)
- [Model server parameters](parameters.md)
