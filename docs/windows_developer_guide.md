# OpenVINO&trade; Model Server Developer Guide for Windows

# Install prerequisites
```
md c:\git
md c:\opt
```

## VISAUL
Visual Studio 2022 with C++ - https://visualstudio.microsoft.com/downloads/

## PYTHON: https://www.python.org/ftp/python/3.9.0/python-3.9.0-amd64.exe in C:\opt\Python39
Python3. (Python 3.11.9 is tested)
pip install numpy==1.23
make sure you install numpy for the python version you pass as build argument
make sure default "python --version" gets you 3.9

## OpenVINO
OpenVINO Runtime: Download 2024.4 https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.4/windows/w_openvino_toolkit_windows_2024.4.0.16579.c3152d32c9c_x86_64.zip
unzipped in c:\opt\intel\openvino_2024

## BAZEL
https://github.com/bazelbuild/bazel/releases/download/6.4.0/bazel-6.4.0-windows-x86_64.exe
https://bazel.build/install/windows -> copy and rename in c:\opt\bazel.exe and add to PATH

## MSYS
Please make sure bash is installed and add its directory in PATH, or --repo_env BAZEL_SH='/path/to/bash'.
Install in c:\opt\msys64 https://github.com/msys2/msys2-installer/releases/download/2024-07-27/msys2-x86_64-20240727.exe

## NPM YARN
https://github.com/coreybutler/nvm-windows/releases/download/1.1.12/nvm-setup.exe
Open command line cmd.exe and run:
```
nvm install latest
nvm use 22.9.0
npm cache clean --force
```

If you want to compile without proxy, npm proxy needs to be reset:
```
set http_proxy=
set https_proxy=
npm config rm https-proxy
npm config rm proxy
npm i --global yarn
yarn
```

## OPENCV install to - "C:\\opt\\opencv\\"
https://github.com/opencv/opencv/releases/download/4.10.0/opencv-4.10.0-windows.exe

## WGET
https://eternallybored.org/misc/wget/1.21.4/64/wget.exe download to c:\opt
Add c:\opt to system env PATH

## Run Developer Command Prompt for VS 2022
## Enable Developer mode on in windows system settings

#### Boring SSL - not needed until md5 hash is needed.
Clone in in c:\opt\
```
git clone https://github.com/firebase/boringSSL-SwiftPM.git
```

## Libevent - ovms link phase
```git clone https://github.com/libevent/libevent
git co release-2.1.12-stable
md build && cd build
C:\git\libevent\build>cmake -G "Visual Studio 17 2022" -DEVENT__DISABLE_OPENSSL=1 -DEVENT_LIBRARY_SHARED=0 ..
cmake --build . --config Release
md c:\opt\libevent
xcopy lib\Release\* c:\opt\libevent\lib\
xcopy /s /e include\event2\* c:\opt\libevent\include\event2\
cd ..
xcopy /s /e include\ c:\opt\libevent\include\
```

# Opencl headers
```
cd c:\opt
git clone https://github.com/KhronosGroup/OpenCL-SDK.git opencl
```

## GET CODE
```
cd C:\git\
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```

## COMPILE
```
bazel build --config=windows --jobs=8 --subcommands --repo_env PYTHON_BIN_PATH=C:/opt/Python39/python.exe --verbose_failures --define CLOUD_DISABLE=1 --define MEDIAPIPE_DISABLE=1 --define PYTHON_DISABLE=1 //src:ovms > compilation.log 2>&1
```

## To run ovms in developer command line
cwd=C:\git\model_server>bazel-out

```
set PATH=%PATH%;C:\opt\intel\openvino_2024\runtime\bin\intel64\Release;C:\Windows\SysWOW64\;C:\opt\intel\openvino_2024\runtime\3rdparty\tbb\bin\;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Redist\MSVC\14.29.30133\debug_nonredist\x64\Microsoft.VC142.DebugCRT\

bazel-out\x64_windows-opt\bin\src\ovms.exe --help
```

## DEPLOY
Open cmd.exe in c:\opt
```
md test\model\1
xcopy /r /s /e /Y C:\git\model_server\bazel-out\x64_windows-opt\bin\src\ovms.exe c:\opt\test
c:\opt\intel\openvino_2024\setupvars.bat
C:\opt\opencv\build\setup_vars_opencv4.cmd
cd c:\opt\test
wget https://www.kaggle.com/api/v1/models/tensorflow/faster-rcnn-resnet-v1/tensorFlow2/faster-rcnn-resnet50-v1-640x640/1/download -O 1.tar.gz
tar xzf 1.tar.gz -C model\1
```

## Start server
```
ovms.exe --model_name faster_rcnn --model_path model --port 9000
```

## Prepare client
Open second cmd.exe terminal
```
cd c:\opt && md client && cd client
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/object_detection/python/object_detection.py
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/object_detection/python/requirements.txt
wget https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/data/dataset_classes/coco_91cl.txt
wget https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg

pip install --upgrade pip
pip install -r requirements.txt
pip install numpy==1.23
python object_detection.py --image coco_bike.jpg --output output.jpg --service_url localhost:9000
```

# OpenVINO&trade; Model Server VS Code setup on Windows 11
Install VS Code and c/c++ Extension

Copy vscode/launch.json, vscode/tasks.json to .vscode directory.

## Example launch.json
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name" : "OVMS RLS",
            "type": "cppvsdbg",
            "request": "launch",
            "program": "${workspaceFolder}/bazel-out/x64_windows-opt/bin/src/ovms.exe",
            "args": ["--help"],
            "cwd": "${workspaceFolder}/bazel-out/x64_windows-opt/bin/src/",
            "environment": [
                {"name": "PATH", "value": "%PATH%;C:\\opt\\intel\\openvino_2024\\runtime\\bin\\intel64\\Release;C:\\Windows\\SysWOW64\\;C:\\opt\\intel\\openvino_2024\\runtime\\3rdparty\\tbb\\bin\\;C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Redist\\MSVC\\14.29.30133\\debug_nonredist\\x64\\Microsoft.VC142.DebugCRT\\"},
            ],
            "externalConsole": true,
            "logging": {
                "moduleLoad": false,
                "trace": true
            },
            "sourceFileMap": {
                "/c/users/rasapala/_bazel_rasapala/tpoq5oxa/execroot/ovms/": "${workspaceFolder}"
            }
        },
        {
            "name" : "OVMS DEBG",
            "type": "cppvsdbg",
            "request": "launch",
            "program": "${workspaceFolder}/bazel-out/x64_windows-dbg/bin/src/ovms.exe",
            "args": ["--help"],
            "cwd": "${workspaceFolder}/bazel-out/x64_windows-dbg/bin/src/",
            "environment": [
                {"name": "PATH", "value": "%PATH%;C:\\opt\\intel\\openvino_2024\\runtime\\bin\\intel64\\Release;C:\\Windows\\SysWOW64\\;C:\\opt\\intel\\openvino_2024\\runtime\\3rdparty\\tbb\\bin\\;C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Redist\\MSVC\\14.29.30133\\debug_nonredist\\x64\\Microsoft.VC142.DebugCRT\\"},
            ],
            "externalConsole": true,
            "logging": {
                "moduleLoad": false,
                "trace": true
            },
            "sourceFileMap": {
                "/c/users/rasapala/_bazel_rasapala/tpoq5oxa/execroot/ovms/": "${workspaceFolder}"
            }
        }
    ]
}
```

## Example tasks.json
```
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build (Debug)",
            "type": "shell",
            "command": "bazel build -c dbg --config=windows --jobs=24 --subcommands --repo_env PYTHON_BIN_PATH=C:/opt/Python39/python.exe --verbose_failures --define CLOUD_DISABLE=1 --define MEDIAPIPE_DISABLE=1 --define PYTHON_DISABLE=1 //src:ovms",
            "group": {
                "kind": "build",
                "isDefault": true
            },
        },
        {
            "label": "Build (Release)",
            "type": "shell",
            "command": "bazel build -c opt --config=windows --jobs=24 --subcommands --repo_env PYTHON_BIN_PATH=C:/opt/Python39/python.exe --verbose_failures --define CLOUD_DISABLE=1 --define MEDIAPIPE_DISABLE=1 --define PYTHON_DISABLE=1 //src:ovms",
            "group": {
                "kind": "build",
                "isDefault": true
            },
        }
    ]
}
```

## Build
Ctrl + Shift + B
Choose Build (Debug) or Build (Release) in the task dropdown.

## Debug
Press "OVMS DEBUG(model_server)" at the bottom left task bar in visual code.
or
Press "OVMS RELEASE(model_server)" at the bottom left task bar in visual code.

Breakpoints are available after building the Debug solution and choosing OVMS Debug task.