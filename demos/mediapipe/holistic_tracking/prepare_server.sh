#!/bin/bash
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# prepare initial folders structure
mkdir -p mediapipe/mediapipe/modules/hand_landmark/
mkdir -p ovms
# copy a text file for hand landmark module
wget -O mediapipe/mediapipe/modules/hand_landmark/handedness.txt https://raw.githubusercontent.com/openvinotoolkit/mediapipe/v2023.3/mediapipe/modules/hand_landmark/handedness.txt
# copy ovms config including a graph definition
cp config_holistic.json ovms/
cp iris_tracking.pbtxt ovms/
cp holistic_tracking.pbtxt ovms/
# download the models
curl https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite -o ovms/face_detection_short_range/1/face_detection_short_range.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite -o ovms/face_landmark/1/face_landmark.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite -o ovms/hand_landmark_full/1/hand_landmark_full.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/hand_recrop.tflite -o ovms/hand_recrop/1/hand_recrop.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/iris_landmark.tflite -o ovms/iris_landmark/1/iris_landmark.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite -o ovms/palm_detection_full/1/palm_detection_full.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite -o ovms/pose_detection/1/pose_detection.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/pose_landmark_full.tflite -o ovms/pose_landmark_full/1/pose_landmark_full.tflite --create-dirs

# convert pose_detection model to the format supported by OV. It will eliminate DENDIFY layer, which is currently not supported by OpenVINO
chmod 777 ovms/pose_detection/1
docker run -w /home/user/workdir  --rm  -v `pwd`/ovms/pose_detection/1:/home/user/workdir ghcr.io/pinto0309/tflite2tensorflow:latest tflite2tensorflow --model_path pose_detection.tflite --flatc_path ../flatc --schema_path ../schema.fbs --output_pb
docker run --rm -w /home/user/workdir -v `pwd`/ovms/pose_detection/1:/home/user/workdir ghcr.io/pinto0309/tflite2tensorflow:latest tflite2tensorflow --model_path pose_detection.tflite --flatc_path ../flatc --schema_path ../schema.fbs --output_no_quant_float32_tflite
# replace the model and drop temporary files
cp ovms/pose_detection/1/saved_model/model_float32.tflite ovms/pose_detection/1/pose_detection.tflite
docker run --rm -w /home/user/workdir -v `pwd`/ovms/pose_detection/1:/home/user/workdir ghcr.io/pinto0309/tflite2tensorflow:latest rm -Rf saved_model pose_detection.json
chmod -R 755 ovms/ 
