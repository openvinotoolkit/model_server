#
# Copyright (c) 2018 Intel Corporation
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

from tensorflow.python.framework import dtypes as dtypes
from tensorflow.python.saved_model.signature_def_utils import \
    build_signature_def
from tensorflow.python.saved_model.utils import build_tensor_info
from tensorflow.python.ops import gen_array_ops
import numpy as np


def _prepare_signature_outputs(names: list, dtype_layer, shape,
                               model_keys):
    """
    Inference Engine Api does not report outputs shapes,
    for signature format consistency with TF Serving theres returned a
    dummy shape==(1,1,1).
    To be fixed in the future.
    """
    signature = {}
    for key, value in model_keys.items():
        if value in names:
            x = gen_array_ops.placeholder(dtype=dtype_layer, shape=shape,
                                          name=value)
            x_tensor_info = build_tensor_info(x)
            signature[key] = x_tensor_info
    return signature


def _prepare_signature_inputs(layers: dict, dtype_layer, model_keys):
    signature = {}
    for key, value in model_keys.items():
        if value in layers.keys():
            x = gen_array_ops.placeholder(dtype=dtype_layer,
                                          shape=layers[value], name=value)
            x_tensor_info = build_tensor_info(x)
            signature[key] = x_tensor_info
    return signature


def prepare_get_metadata_output(inputs, outputs, model_keys):
    dtype_model = dtypes.as_dtype(np.float32)
    inputs_signature = _prepare_signature_inputs(layers=inputs,
                                                 dtype_layer=dtype_model,
                                                 model_keys=model_keys
                                                 ['inputs'])
    dummy_shape = (1, 1, 1)
    outputs_signature = _prepare_signature_outputs(names=outputs,
                                                   dtype_layer=dtype_model,
                                                   shape=dummy_shape,
                                                   model_keys=model_keys
                                                   ['outputs'])

    signature_def = build_signature_def(inputs_signature, outputs_signature,
                                        "tensorflow/serving/predict")
    return signature_def
