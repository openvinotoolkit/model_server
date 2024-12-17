# OpenVINO&trade; Model Server Deployment Guide for Windows 11 (Experimental/Alpha) {#ovms_docs_windows_binary_guide}
This document describes experimental/alpha windows deployment for ovms.exe binary.
Follow the instructions once you have compiled ovms.exe and you want to deploy the binary on a Windows 11 system.
[Developer Guide for Windows](windows_developer_guide.md)

OpenVINO&trade; Model Server is in experimental/alpha stage of windows enabling with limited functionality and quality.
It is recommended to use the top of main repository branch for more feature enabled code and better software quality for windows.

## List of enabled features:
### Limited model server basic functionality besides disabled features
### GRPC API
### Mediapipe graphs execution
### Serving single models in all formats

## List of disabled features:
### Ovms feature parity with Linux implementation
### LLM support
### PYTHON NODES support
### REST API support
### Custom nodes support
### Cloud storage support
### Model cache support
### DAG pipelines

# Install prerequisites
```
md c:\git
md c:\opt
```

## Microsoft Visual C++ Redistributable
Install C++ package - https://aka.ms/vs/17/release/VC_redist.x64.exe

## PYTHON: https://www.python.org/ftp/python/3.9.0/python-3.9.0-amd64.exe in C:\opt\Python39
Python3.9
```
pip install numpy==1.23
```
make sure you install numpy for the python version you pass as build argument
make sure default "python --version" gets you 3.9

## Install dependencies
Open cmd.exe and run windows_install_dependencies.bat
```
windows_install_dependencies.bat
```

## DEPLOY
Open cmd.exe in c:\opt
```
md test\model\1
C:\opt\openvino\setupvars.bat
C:\opt\opencv\setup_vars_opencv4.cmd
xcopy /r /Y ovms.exe c:\opt\test
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
cd c:\opt
md client
cd client
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/object_detection/python/object_detection.py
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/object_detection/python/requirements.txt
wget https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/data/dataset_classes/coco_91cl.txt
wget https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg

pip install --upgrade pip
pip install -r requirements.txt
pip install numpy==1.23
```

## Run client inference
```
python object_detection.py --image coco_bike.jpg --output output.jpg --service_url localhost:9000
```