#
# Copyright (c) 2025 Intel Corporation
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

import openvino.runtime as ov
import numpy as np
import os

batch_dim = []
shape = [1, 10]
dtype = np.int8
model_name = "no_name_output"
model_version_dir = model_name
print(batch_dim + shape)
in0 = ov.opset1.parameter(shape=batch_dim + shape, dtype=dtype, name="input_1")
in1 = ov.opset1.parameter(shape=batch_dim + shape, dtype=dtype, name="input_2")
op0 = ov.opset1.multiply(in1, in0, name="MULTIPLY")
op1 = ov.opset1.add(in1, in0, name="ADD")

model = ov.Model([op0, op1], [in0, in1], model_name)

for idx, inp in enumerate(model.inputs):  
    print(f"Input {idx}: {inp.get_names()} {inp.get_shape()} {inp.get_index()}")
print(model.outputs)
for idx, out in enumerate(model.outputs):  
    print(f"Output {idx}: {out.get_names()} {out.get_shape()} {out.get_index()} ")
    assert len(out.get_names()) == 0, "number of output names should be 0"

try:
    os.makedirs(model_version_dir)
except OSError as ex:
    pass  # ignore existing dir

ov.serialize(model, model_version_dir + "/model.xml", model_version_dir + "/model.bin")

ov_model = ov.Core().read_model(model_version_dir + "/model.xml")
compiled_model = ov.Core().compile_model(model, "CPU")

input_data = np.ones((1, 10),dtype=np.int8)*10
results = compiled_model({"input_1": input_data, "input_2": input_data})
assert np.all(results[0] == 100), "for inputs np.ones((1, 10), the expected output is 100 in every element: 10*10"
assert np.all(results[1] == 20), "for inputs np.ones((1, 10), the expected output is 20 in every element: 10+10"

