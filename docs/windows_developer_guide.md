# OpenVINO&trade; Model Server Developer Guide for Windows
This document describes windows development and compilation guide for ovms.exe binary.
This instruction was tested on Windows 11 and Windows 10 OS.

## List of disabled features on Windows model server:
- cloud storage for the models - to be added in next releases
- C-API interface - to be added in next releases
- DAG pipelines - legacy feature


# Install prerequisites
Following the steps below requires 40GB of free disk space.

## VISUAL BUILD TOOLS
Install build tools for VS:

https://aka.ms/vs/17/release/vs_BuildTools.exe

Mark required options for installation:
- C++ Desktop development with C++
- Windows 11 SDK (10.0.26210.0)
- MSVC v143 CPP - VS 2022 C++ platform toolset.
- C++ CMake tools for Windows platform toolset.
- MSVC v142 CPP - VS 2022 C++ platform toolset.
- Optional Windows 11 SDK (10.0.26100.0) for Windows 10 compilation

![Build Tools options](build_tools.jpg)

## Power shell settings
Set Execution Policy to RemoteSigned
Open PowerShell as an administrator: Right-click on the Start button and select “Windows PowerShell (Admin)”.
Run the command:
```Set-ExecutionPolicy Unrestricted -Scope CurrentUser -Force```

## Enable Developer mode in windows system settings
Follow instructions in the link below:
https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development

## Run Command Prompt
Press Windows Start and run the cmd.exe terminal as Administrator.
Run commands in this prompt is not stated otherwise.

## Pull OpenVINO Model Server source
> `Git` is required to complete this step. If you don't have it on your system, download it from https://git-scm.com/downloads/win and install before you continue.
Run below commands in terminal to clone model server repository:
```bat
mkdir C:\git
cd C:\git\
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```

## Install dependencies
Run windows_install_build_dependencies.bat
This will install around 6 GB dependencies in the c:\opt directory:
- wet.exe, msys2 tools, Openvinotoolkit for GenAI, OpenCL headers, BoringSSL, bazel, Python 3.12.10, OpenCV, Curl

If error occurs during the script execution, please fix the error and rerun the script.
```bat
windows_install_build_dependencies.bat
```

Optionally, you add parameter to the windows_install_build_dependencies.bat script
```bat
windows_install_build_dependencies.bat my_dir_on_c 1 1
```
[arg1] - This way you can change default dependency install directory to c:\my_dir_on_c
[arg2] - Set the clean flag to 0 or 1 - this will cleanup the installation directories and reinstall all dependencies
[arg3] - Add the compilation integrity flag to 0 or 1 - set the additional integritycheck and Qspectre compilation flag when compiling dependencies

## COMPILE
[WARNING] This step consumes up to 13GB of disk space. It can take up to 1h depending on host CPU and internet connection speed.
```bat
windows_build.bat
```

Optionally, you add parameter to the windows_build.bat script
```bat
windows_build.bat my_dir_on_c --with_tests --integrity
```
[arg1] - Change the default dependency location to `C:\my_dir_on_c`.
[arg2..3] - Optional `--with_tests` and `--integrity` flags. They can be supplied individually or in either order.

# Running unit tests - optional
The script builds and runs the Python-enabled `ovms_test` binary, the Python runtime library test, and the Python interpreter lifecycle, backend-failure, and borrowed-interpreter tests. It also runs the Windows service tests and prepares the LLM test models (src\test\llm_testing).
```bat
windows_test.bat
```

Optionally, pass the dependency location and a GoogleTest filter:
```bat
windows_test.bat my_dir_on_c PythonFlowTest.*
```
[arg1] - Dependency location under `C:\`; defaults to `C:\opt`.
[arg2] - GoogleTest filter for `ovms_test`; defaults to `*`.

To skip downloading the Hugging Face test models, use the built-in negative
filter for tests that do not access those models:
```bat
windows_test.bat opt --skip-model-download
```
The filter is applied only to `ovms_test`; the Windows service and Python
runtime tests still run. To provide a custom model-independent filter, pass it
before the option:
```bat
windows_test.bat opt "ModelVersionStatus.*" --skip-model-download
```

# Creating deployment package
This step prepares ovms.zip deployment package from the build artifacts in the dist\windows\ directory. Run this script after successful compilation.
```bat
windows_create_package.bat
```

To include the embedded Python runtime, pass `--with_python` after the dependency location:
```bat
windows_create_package.bat my_dir_on_c --with_python
```
[arg1] - Dependency location under `C:\`; defaults to `C:\opt`.
[arg2] - `--with_python` includes the embedded Python runtime.

# Test the Deployment
You can follow the [baremetal deployment guide](deploying_server_baremetal.md) for information how to deploy and use the ovms.zip package.

# Developer Command Prompt
For building ovms.exe and running ovms_test.exe with manual bazel commands you must setup proper environment variables.
Run the batch script in new "Developer Command Prompt for VS 2022" terminal:
```bat
cd c:\git\model_server
windows_setupvars.bat
```

Build the unit-test and Python runtime test targets with the same configuration used by `windows_test.bat`:
```bat
bazel --output_user_root=c:\opt build --config=win_mp_on --action_env OpenVINO_DIR=c:\opt/openvino/runtime/cmake --jobs=%NUMBER_OF_PROCESSORS% --verbose_failures //src:ovms_test //src:python_runtime_library_test //src/python:pythoninterpretermodule_lifecycle_test //src/python:pythoninterpretermodule_backend_failure_test //src/python:pythoninterpretermodule_borrowed_interpreter_test 2>&1 | tee win_build_test.log
```

Download LLMs
```bat
%cd%\windows_prepare_llm_models.bat %cd%\src\test\llm_testing
```

Change tests configs to windows:
```bat
python windows_change_test_configs.py
```

Run specific unit tests by setting gtest_filter:
```bat
%cd%\bazel-bin\src\ovms_test.exe --gtest_filter=* 2>&1 | tee win_full_test.log
```

Run the Python runtime library test:
```bat
%cd%\bazel-bin\src\python_runtime_library_test.exe --gtest_filter=*
```

Run the Python interpreter module tests:
```bat
bazel --output_user_root=c:\opt test --config=win_mp_on --action_env OpenVINO_DIR=c:\opt/openvino/runtime/cmake --jobs=%NUMBER_OF_PROCESSORS% --test_output=streamed --test_filter=* //src/python:pythoninterpretermodule_lifecycle_test //src/python:pythoninterpretermodule_backend_failure_test //src/python:pythoninterpretermodule_borrowed_interpreter_test
```
