#!/bin/bash
#
# Copyright (c) 2021 Intel Corporation
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

# Variables setup
export KALDI_PATH=/opt/kaldi
export ASPIRE_PATH=$KALDI_PATH/egs/aspire/s5
export DATA_PATH=/opt/data

# Model download and unpack 
cd $ASPIRE_PATH
wget https://kaldi-asr.org/models/1/0001_aspire_chain_model_with_hclg.tar.bz2
tar -xvf 0001_aspire_chain_model_with_hclg.tar.bz2

apt-get install -y virtualenv
cd /opt/model_server
virtualenv -p python3 .venv 
. .venv/bin/activate 
pip install tensorflow-serving-api==2.* kaldi-python-io==1.2.1