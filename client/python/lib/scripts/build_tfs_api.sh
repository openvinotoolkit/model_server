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

git clone https://github.com/tensorflow/tensorflow.git tf
git clone https://github.com/tensorflow/serving.git tfs

protoc --proto_path=$PWD/tfs --proto_path=$PWD/tf --python_out=$PWD \
$PWD/tf/tensorflow/core/framework/*.proto \
$PWD/tf/tensorflow/core/example/*.proto \
$PWD/tf/tensorflow/core/protobuf/*.proto \
$PWD/tfs/tensorflow_serving/util/*.proto \
$PWD/tfs/tensorflow_serving/config/*.proto \
$PWD/tfs/tensorflow_serving/core/*.proto \
$PWD/tfs/tensorflow_serving/apis/*.proto

cp tfs/tensorflow_serving/apis/prediction_service_pb2_grpc.py tensorflow_serving/apis/.
cp tfs/tensorflow_serving/apis/model_service_pb2_grpc.py tensorflow_serving/apis/.
rm -rf tf/ tfs/
