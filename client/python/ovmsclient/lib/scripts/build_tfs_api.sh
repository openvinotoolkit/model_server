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

set -e

cleanup_tmp_dirs() {
    ARG=$?
    echo "Cleaning up temp directories"
    rm -rf tf/ tfs/ compiled_protos/
    exit $ARG
}

trap cleanup_tmp_dirs EXIT

git clone --branch v2.5.0 --depth 1 https://github.com/tensorflow/tensorflow.git tf
git clone --branch 2.5.1 --depth 1 https://github.com/tensorflow/serving.git tfs

rm -rf ovmsclient/tfs_compat/protos compiled_protos
mkdir -p ovmsclient/tfs_compat/protos
mkdir compiled_protos
cp -R tf/tensorflow ovmsclient/tfs_compat/protos/tensorflow
cp -R tfs/tensorflow_serving ovmsclient/tfs_compat/protos/tensorflow_serving 
find ovmsclient/tfs_compat/protos -name "*.proto" -exec sed -i -E 's/import "tensorflow/import "ovmsclient\/tfs_compat\/protos\/tensorflow/g' {} \;
python3 scripts/rename_proto_package.py

protoc --proto_path=$PWD --python_out=$PWD/compiled_protos \
$PWD/ovmsclient/tfs_compat/protos/tensorflow/core/framework/*.proto \
$PWD/ovmsclient/tfs_compat/protos/tensorflow/core/example/*.proto \
$PWD/ovmsclient/tfs_compat/protos/tensorflow/core/protobuf/*.proto \
$PWD/ovmsclient/tfs_compat/protos/tensorflow_serving/util/*.proto \
$PWD/ovmsclient/tfs_compat/protos/tensorflow_serving/config/*.proto \
$PWD/ovmsclient/tfs_compat/protos/tensorflow_serving/core/*.proto \
$PWD/ovmsclient/tfs_compat/protos/tensorflow_serving/apis/*.proto

sed -i 's/from tensorflow_serving.apis/from ovmsclient.tfs_compat.protos.tensorflow_serving.apis/g' ovmsclient/tfs_compat/protos/tensorflow_serving/apis/prediction_service_pb2_grpc.py
sed -i 's/from tensorflow_serving.apis/from ovmsclient.tfs_compat.protos.tensorflow_serving.apis/g' ovmsclient/tfs_compat/protos/tensorflow_serving/apis/model_service_pb2_grpc.py

cp ovmsclient/tfs_compat/protos/tensorflow_serving/apis/prediction_service_pb2_grpc.py compiled_protos/ovmsclient/tfs_compat/protos/tensorflow_serving/apis/.
cp ovmsclient/tfs_compat/protos/tensorflow_serving/apis/model_service_pb2_grpc.py compiled_protos/ovmsclient/tfs_compat/protos/tensorflow_serving/apis/.

rm -rf ovmsclient/tfs_compat/protos
cp -R compiled_protos/ovmsclient/tfs_compat/protos ovmsclient/tfs_compat/protos

