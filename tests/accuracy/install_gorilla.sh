#!/bin/bash -x
#
# Copyright (c) 2026 Intel Corporation
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

BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
# install dependencies
# if gorilla is already installed, it will skip installation
if [ ! -d "gorilla" ]; then
    git clone https://github.com/ShishirPatil/gorilla
    cd gorilla/berkeley-function-call-leaderboard
    git checkout 9b8a5202544f49a846aced185a340361231ef3e1
    curl -s https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/${BRANCH_NAME}/demos/continuous_batching/accuracy/gorilla.patch | git apply -v
    pip install -e . --extra-index-url "https://download.pytorch.org/whl/cpu"
    bfcl --help
    cd ../..
    cp test_case_ids_to_generate.json gorilla/berkeley-function-call-leaderboard/
else
    echo "Gorilla already installed, skipping installation. Delete the 'gorilla' directory to reinstall."
fi