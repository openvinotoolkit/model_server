# OpenVINO&trade; Model Server Developer Guide for Windows
This document describes windows development and compilation guide for ovms.exe binary.
This instruction was tested on Windows 11 and Windows 10 OS.

## List of disabled features:
### Cloud storage support

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
Install git from https://git-scm.com/downloads/win
After that run commands in terminal:
```bat
mkdir C:\git
cd C:\git\
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```

## Install dependencies
Run windows_install_build_dependencies.bat
This will install around 3.3 GB dependencies in the c:\opt directory:
- wet.exe, msys2 tools, Openvinotoolkit, OpenCL headers, BoringSSL, bazel, Python 3.11.9, OpenCV

If error occurs during the script execution, please fix the error and rerun the script.
```bat
windows_install_build_dependencies.bat
```

## COMPILE
[WARNING] This step consumes 11GB of disk space. It can take up to 1h depending on host CPU and internet connection speed.
```bat
windows_build.bat
```

# Running unit tests - optional
The script compiles ovms_test binary, downloads and converts test LLM models (src\tests\llm_testing) and installs Python torch and optimum.
```bat
windows_test.bat
```

# Creating deployment package
This step prepares ovms.zip deployment package from the build artifacts in the dist\windows\ directory. Run this script after successful compilation.
```bat
windows_create_package.bat
```

# Test the Deployment
You can follow the [baremetal deployment guide](deploying_server_baremetal.md) for information how to deploy and use the ovms.zip package.
