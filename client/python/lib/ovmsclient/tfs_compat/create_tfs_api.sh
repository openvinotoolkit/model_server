#!/bin/bash

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
