#
# Copyright (c) 2018-2020 Intel Corporation
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

from tensorflow import __version__ as tf_version
from tensorflow.python.saved_model.signature_def_utils import \
    build_signature_def
from tensorflow.python.saved_model.utils import build_tensor_info
if tf_version.split(".")[0] == "2":
    from tensorflow.python.ops import array_ops
    from tensorflow.python.framework.ops import disable_eager_execution
else:  # TF version 1.x
    from tensorflow.python.ops import gen_array_ops as array_ops

type_mapping = {
    'FP32': 1,
    'FP16': 0,
    'I8': 9,
    'I32': 3,
    'I16': 8,
    'U32': 6,
    'U16': 5
}
# mapping supported precisions from https://github.com/opencv/dldt/blob/2018/
# inference-engine/ie_bridges/python/inference_engine/ie_api.pyx with TF types
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/
# framework/dtypes.py


def _prepare_signature(layers: dict, model_keys):
    if tf_version.split(".")[0] == "2":
        disable_eager_execution()
    signature = {}
    for key, value in model_keys.items():
        if value in layers.keys():
            x = array_ops.placeholder(
                dtype=type_mapping[layers[value].precision],
                shape=layers[value].shape, name=value)
            x_tensor_info = build_tensor_info(x)
            signature[key] = x_tensor_info
    return signature


def prepare_get_metadata_output(inputs, outputs, model_keys):
    inputs_signature = _prepare_signature(
        layers=inputs, model_keys=model_keys['inputs'])
    outputs_signature = _prepare_signature(
        layers=outputs, model_keys=model_keys['outputs'])

    signature_def = build_signature_def(inputs_signature, outputs_signature,
                                        "tensorflow/serving/predict")
    return signature_def
