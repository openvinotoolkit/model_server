#!/bin/bash -x
#
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
if [ -d "$1" ]; then
  echo "Models directory $1 exists. Skipping downloading models."
  exit 0
fi
mkdir -p $1
wget -O $1/face-detection-adas-0001.xml https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-adas-0001/FP32/face-detection-adas-0001.xml
wget -O $1/face-detection-adas-0001.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-adas-0001/FP32/face-detection-adas-0001.bin

