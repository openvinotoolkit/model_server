#
# Copyright (c) 2022 Intel Corporation
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
from ubuntu:22.04

env DEBIAN_FRONTEND=noninteractive
run apt-get update && apt-get install -y git software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
python3.8 python3.9 python3.10 python3.11 python3.12 python3.13 \
python3.8-distutils python3.9-distutils python3.10-distutils python3.11-distutils \
python3.12-dev python3.13-dev \
python3.8-venv python3.9-venv python3.10-venv python3.11-venv python3.12-venv python3.13-venv \
python3-pip
run pip install invoke
copy client /model_server/client
copy demos /model_server/demos
copy tests/python/tasks.py .
run invoke deps --verbose --fastFail
entrypoint ["invoke", "test"]
