## Deploying Model Server on Baremetal {#ovms_docs_deploying_server_baremetal}

OpenVINO Model Server can run outside a container using precompiled packages for
Windows 11, Ubuntu 24.04, Ubuntu 26.04, and Red Hat Enterprise Linux 9.

Starting from version 2026.5, there is **one package per operating system**. Separate `python_on` and
`python_off` packages are no longer provided. Python integration is optional:
the same package can run without Python, or enable Python features when a
compatible Python installation and its dependencies are available.

The instructions below use the [latest weekly packages](https://storage.openvinotoolkit.org/repositories/openvino_model_server/packages/weekly/latest/).
Weekly packages are built from the main branch and let you evaluate features
ahead of official releases. The contents of `latest` change with each weekly build.

## Install and Verify

Download and extract the package for your operating system. Run the executable
directly from the extracted directory.

::::{tab-set}
:::{tab-item} Windows
:sync: windows

Install the [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/VC_redist.x64.exe).

Download, extract, and verify the package in Command Prompt or PowerShell:

```bat
curl.exe -fL http://ov-share-13.sclab.intel.com/cv_bench_cache/OVMS_do_not_remove/ovms_artefacts/2026.5.0/test/ovms_windows_2026.5.0.zip -o ovms_windows.zip
tar -xf ovms_windows.zip
.\ovms\ovms.exe --version
```

The Windows package includes an embedded Python runtime and packages in the
`ovms\python` subfolder; a separate Python installation is not required.
You can delete this subfolder if Python capabilities are not needed.

Optionally, install OVMS as a [Windows service](windows_service.md):

```bat
.\ovms\install_ovms_service.bat
```

:::
:::{tab-item} Ubuntu 24.04
:sync: ubuntu-24-04

Install system dependencies, then download, extract, and verify the package:

```bash
sudo apt update
sudo apt install -y curl ca-certificates libxml2
curl -fLO http://ov-share-13.sclab.intel.com/cv_bench_cache/OVMS_do_not_remove/ovms_artefacts/2026.5.0/test/ovms_ubuntu24_2026.5.0_test.tar.gz
tar -xzf ovms_ubuntu24_2026.5.0_test.tar.gz
./ovms/bin/ovms --version
```

:::
:::{tab-item} Ubuntu 26.04
:sync: ubuntu-26-04

Install system dependencies, then download, extract, and verify the package:

```bash
sudo apt update
sudo apt install -y curl ca-certificates libxml2-16
curl -fLO http://ov-share-13.sclab.intel.com/cv_bench_cache/OVMS_do_not_remove/ovms_artefacts/2026.5.0/test/ovms_ubuntu26_2026.5.0_test.tar.gz
tar -xzf ovms_ubuntu26_2026.5.0_test.tar.gz
./ovms/bin/ovms --version
```

:::
:::{tab-item} Red Hat Enterprise Linux 9
:sync: rhel-9

Install system dependencies, then download, extract, and verify the package:

```bash
sudo dnf install -y curl ca-certificates libxml2
curl -fLO http://ov-share-13.sclab.intel.com/cv_bench_cache/OVMS_do_not_remove/ovms_artefacts/2026.5.0/test/ovms_redhat_2026.5.0_test.tar.gz
tar -xzf ovms_redhat_2026.5.0_test.tar.gz
./ovms/bin/ovms --version
```

:::
::::

## Optional Python Integration

Python integration provides two features:

- A **Jinja-based chat-template engine for LLM models**, using Jinja2 to render
  model chat templates.
- **Python custom nodes in MediaPipe graphs**, allowing graphs to execute
  user-defined Python code.

Python is not required to start OVMS or serve models that do not use these
features. On Linux, install the compatible Python runtime for your package.
On Windows, use the embedded runtime in the `ovms\python` subfolder, which
already includes **Jinja2** and **MarkupSafe**. On Linux, install Jinja2 for
chat-template rendering before starting OVMS; its MarkupSafe dependency is
installed automatically.

::::{tab-set}
:::{tab-item} Windows
:sync: windows

The embedded Python runtime already includes Jinja2 and MarkupSafe. No separate
Python installation or installation of these packages is required.
Keep the `ovms\python` subfolder
when using either Python feature; delete it only if neither feature is needed.

:::
:::{tab-item} Ubuntu 24.04
:sync: ubuntu-24-04

Install Python 3.12, its shared library, and Jinja2 from the system packages:

```bash
sudo apt install -y python3 libpython3.12 python3-jinja2
```

:::
:::{tab-item} Ubuntu 26.04
:sync: ubuntu-26-04

Install Python 3.14, its shared library, and Jinja2 from the system packages:

```bash
sudo apt install -y python3 libpython3.14 python3-jinja2
```

:::
:::{tab-item} Red Hat Enterprise Linux 9
:sync: rhel-9

Install Python 3.12 and pip, then install Jinja2 for the current user:

```bash
sudo dnf install -y python3.12 python3.12-libs python3.12-pip
python3.12 -m pip install --user Jinja2
```

:::
::::

If your [Python nodes](./python_support/reference.md) use OpenVINO or OpenVINO
GenAI, also install NumPy for the same interpreter (`python3-numpy` on Ubuntu,
`python3.12 -m pip install --user numpy` on Red Hat, or
`.\ovms\python\python.exe -m pip install numpy` on Windows).

**Do not install `openvino`, `openvino-tokenizers`, or `openvino-genai` via pip
for OVMS usage.** These packages are already included in the OVMS package;
installing other versions can cause dependency conflicts.

If a compatible Python runtime or required Python plugins are unavailable,
OVMS continues with Python features disabled. Python nodes are then unavailable,
and LLM chat-template rendering has reduced functionality. Enable Python
integration for advanced templates and tool-calling scenarios. Python support
does not guarantee that every model's chat template will render successfully.

For runtime diagnostics, see [Python Runtime Setup and Troubleshooting](python_runtime_setup_and_troubleshooting.md).

## Test the Deployment

Download a ResNet50 model (use `curl.exe` instead of `curl` in Windows PowerShell):

```console
curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.bin -O
curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.xml -O
```

Start the server:

```console
ovms --model_name resnet --model_path resnet50.xml --mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" --port 9000

```

You can also run the server as a background process, a system service on Linux,
or a Windows service, depending on your hosting requirements.

Learn more about Model Server [starting parameters](parameters.md).

> **NOTE**:
> When serving models on [AI accelerators](accelerators.md), additional device
> drivers and dependencies may be required. See
> [Additional Configurations for Hardware](https://docs.openvino.ai/2026/get-started/install-openvino/configurations.html).

## Next Steps

- [Start the server](starting_server.md)
- Try the Model Server [features](features.md)
- Explore the Model Server [demos](../demos/README.md)

## Additional Resources

- [Preparing Model Repository](models_repository.md)
- [Using Cloud Storage](using_cloud_storage.md)
- [Python Runtime Setup and Troubleshooting](python_runtime_setup_and_troubleshooting.md)
- [Troubleshooting](troubleshooting.md)
- [Model Server parameters](parameters.md)
