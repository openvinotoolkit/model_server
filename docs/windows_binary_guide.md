# OpenVINO&trade; Model Server Developer Guide for Windows
## No MEDIAPIPE support
## No PYTHON NODES support
## No REST API support

# Install prerequisites
```
md c:\git
md c:\opt
```

## VISAUL
Visual Studio 2022 with C++ - https://visualstudio.microsoft.com/downloads/

## PYTHON: https://www.python.org/ftp/python/3.9.0/python-3.9.0-amd64.exe in C:\opt\Python39
Python3. (Python 3.11.9 is tested)
```
pip install numpy==1.23
```
make sure you install numpy for the python version you pass as build argument
make sure default "python --version" gets you 3.9

## OpenVINO
OpenVINO Runtime: Download 2024.4 https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.4/windows/w_openvino_toolkit_windows_2024.4.0.16579.c3152d32c9c_x86_64.zip
unzipped in /opt/intel/openvino_2024

## OPENCV install to - "C:\\opt\\opencv\\"
https://github.com/opencv/opencv/releases/download/4.10.0/opencv-4.10.0-windows.exe

## WGET
https://eternallybored.org/misc/wget/1.21.4/64/wget.exe download to c:\opt
Add c:\opt to system env PATH

## DEPLOY
Open cmd.exe in c:\opt
```
md test\model\1
C:\opt\intel\openvino_2024\setupvars.bat
C:\opt\opencv\build\setup_vars_opencv4.cmd
xcopy /r /s /e /Y ovms.exe c:\opt\test
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