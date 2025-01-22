# OpenVINO&trade; Model Server Developer Guide for Windows
This document describes windows development and compilation guide for ovms.exe binary.
This solution was tested on Windows 11.

## List of disabled features:
### Cloud storage support

# Install prerequisites

## VISUAL BUILD TOOLS
Install build tools for VS:

https://aka.ms/vs/17/release/vs_BuildTools.exe

Mark required options for installation:
- C++ Desktop development with C++
- Windows 11 SDK
- MSVC v143 CPP - VS 2022 C++ platform toolset.
- C++ CMake tools for Windows platform toolset.
- MSVC v142 CPP - VS 2022 C++ platform toolset.

![Build Tools options](build_tools.jpg)

## Power shell settings
Set Execution Policy to RemoteSigned
Open PowerShell as an administrator: Right-click on the Start button and select “Windows PowerShell (Admin)”.
Run the command:
```Set-ExecutionPolicy Unrestricted -Scope CurrentUser -Force```
Confirm the change by typing “A” and pressing Enter.

## Enable Developer mode in windows system settings
Follow instructions in the link below:
https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development

## Run Developer Command Prompt for VS 2022
Press Windows Start and paste in search bar "Developer Command Prompt for VS 2022" to open command interpreter windows for VS C++ developers
Run commands in this prompt is not stated otherwise.

# Building without proxy
Please set the proxy setting for windows for in environment variables when building behind proxy
```bat
set HTTP_PROXY=
set HTTPS_PROXY=
```
Also remove proxy from your .gitconfig

## Building with proxy
Please set the proxy setting for windows for in environment variables when building behind proxy
```bat
set HTTP_PROXY=my.proxy.com:123
set HTTPS_PROXY=my.proxy.com:122
```

## NPM YARN
Download and run the nvm installer.
https://github.com/coreybutler/nvm-windows/releases/download/1.1.12/nvm-setup.exe
After installation run below commands,
Run in command line:
```bat
nvm install 22.9.0
nvm use 22.9.0
npm cache clean --force
```

If you want to compile without proxy, npm proxy needs to be reset:
```bat
set http_proxy=
set https_proxy=
npm config rm https-proxy
npm config rm proxy
npm i --global yarn
yarn
```

## GET CODE
```bat
mkdir C:\git
cd C:\git\
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```

## Install dependencies
Run windows_install_dependencies.bat
This will install around 3.3 GB dependencies in the c:\opt directory:
- wet.exe, msys2 tools, Openvinotoolkit, OpenCL headers, BoringSSL, bazel, Python 3.11.9, OpenCV
```bat
windows_install_dependencies.bat
```

## COMPILE

After installation of build dependencies, ovms compilation should be started in new shell "Developer Command Prompt for VS 2022":
[WARNING] It can take up to 1h depending on host CPU and internet connection speed.
```
windows_build.bat
```

# Running unit tests - optional
The script compiles ovms_test binary, downloads and converts test LLM models and installs Python torch and optimum.
```
windows_test.bat
```

# Creating deployment package
This step prepares ovms.zip deployment package from the build artifacts in the dist\windows\ directory. Run this script after successful compilation.
```
windows_create_package.bat
```

# Test the Deployment
You can follow the [baremetal deployment guide](deploying_server_baremetal.md) for information how to deploy and use the ovms.zip package.
