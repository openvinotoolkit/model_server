#!/bin/bash
# Copyright (c) 2024 Intel Corporation
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
mkdir -p ovms/irisTracking
# copy ovms config including a graph definition
cp config_iris.json ovms/
cp iris_tracking.pbtxt ovms/irisTracking
# download the models
curl https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite -o ovms/face_detection_short_range/1/face_detection_short_range.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite -o ovms/face_landmark/1/face_landmark.tflite --create-dirs
curl https://storage.googleapis.com/mediapipe-assets/iris_landmark.tflite -o ovms/iris_landmark/1/iris_landmark.tflite --create-dirs
chmod 755 ovms/ -R
