#!/bin/bash
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

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
onnx_dir=${COMPOSER_ONNX_DIR:-$repo_root/demos/using_onnx_model/python/workspace/resnet50-onnx/1}
paddle_demo="$repo_root/demos/classification_using_paddlepaddle_model/python"
paddle_dir=${COMPOSER_PADDLE_DIR:-$paddle_demo/model/1}
onnx_model="$onnx_dir/resnet50-caffe2-v1-9.onnx"
stage=$(mktemp -d)
onnx_download=
trap 'rm -rf -- "$stage"; if [[ -n $onnx_download ]]; then rm -f -- "$onnx_download"; fi' EXIT

mkdir -p "$onnx_dir" "$paddle_dir"
if [[ ! -s $onnx_model ]]; then
  onnx_download=$(mktemp "$onnx_dir/.resnet50.XXXXXX")
  curl --fail --location --connect-timeout 10 --max-time 120 \
    https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx \
    -o "$onnx_download"
  test -s "$onnx_download"
  mv -- "$onnx_download" "$onnx_model"
  onnx_download=
else
  echo "Reusing ONNX model in $onnx_dir"
fi
if [[ ! -s $paddle_dir/inference.pdmodel || ! -s $paddle_dir/inference.pdiparams ]]; then
  (cd "$stage" && timeout 120s python3 "$paddle_demo/download_model.py")
  install -m 644 "$stage/model/1/inference.pdmodel" "$stage/model/1/inference.pdiparams" "$paddle_dir/"
else
  echo "Reusing PaddlePaddle model in $paddle_dir"
fi
chmod a+rx "$onnx_dir" "$(dirname "$onnx_dir")" "$paddle_dir" "$(dirname "$paddle_dir")"
chmod a+r "$onnx_model" "$paddle_dir/inference.pdmodel" "$paddle_dir/inference.pdiparams"
