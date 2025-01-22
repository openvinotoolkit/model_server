# OpenVINO&trade; Model Server Developer Guide for Windows {#ovms_docs_windows_developer_guide}
This document describes windows development and compilation guide for ovms.exe binary.

## List of disabled features:
### Cloud storage support

# Install prerequisites

## Power shell settings
Set Execution Policy to RemoteSigned
Open PowerShell as an administrator: Right-click on the Start button and select “Windows PowerShell (Admin)”.
Run the command:
```Set-ExecutionPolicy RemoteSigned```
Confirm the change by typing “A” and pressing Enter.

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
```bat
windows_install_dependencies.bat
```

## COMPILE

For building and running ovms.exe after the windows_install_dependencies.bat was successful run the batch script in new "Developer Command Prompt for VS 2022":
```
windows_build.bat
```

# Running unit tests - optional
```
windows_test.bat
```

# Creating deployment package
This step prepares ovms.zip package in the dist\windows\ directory
```
windows_create_package.bat
```

# Deploying ovms
Copy and unpack model server archive for Windows:

```bat
tar -xf ovms.zip
```

Run `setupvars` script to set required environment variables. 

**Windows Command Line**
```bat
./ovms/setupvars.bat
```

**Windows PowerShell**
```powershell
./ovms/setupvars.ps1
```

# Test the Deployment
Download ResNet50 model:
```console
mkdir models/resnet50/1

curl -k https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o models/resnet50/1/model.xml
curl -k https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin -o models/resnet50/1/model.bin
```

Start the server:
```console
ovms --model_name resnet --model_path models/resnet50
```

## **Model Server deployment**: You can find more information on using ovms here: [baremetal deployment guide](deploying_server_baremetal.md)