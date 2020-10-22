#!/bin/bash -x
curl --create-dirs https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.xml https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o model/face-detection-retail-0004.xml -o model/face-detection-retail-0004.bin
curl --create-dirs https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/images/people/people1.jpeg -o images/people1.jpeg
chmod 777 -vR ./images/ ./model/
