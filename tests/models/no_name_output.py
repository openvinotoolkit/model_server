import openvino.runtime as ov
import numpy as np
import os
batch_dim = []
shape = [1, 10]
dtype = np.int32
model_name = "no_name_output"
model_version_dir = model_name
print(batch_dim + shape)
in0 = ov.opset1.parameter(shape=batch_dim + shape, dtype=dtype, name="INPUT1")
in1 = ov.opset1.parameter(shape=batch_dim + shape, dtype=dtype, name="INPUT2")
#tmp1 = ov.opset1.add(in0, start)
#tmp2 = ov.opset1.multiply(end, corrid)
#tmp = ov.opset1.add(tmp1, tmp2)
op0 = ov.opset1.multiply(in1, in0, name="MULTIPLY")
op1 = ov.opset1.add(in1, in0, name="ADD")

# op0 = (input+start+(end*corrid))*ready

model = ov.Model([op0, op1], [in0, in1], model_name)

for idx, inp in enumerate(model.inputs):  
    print(f"Input {idx}: {inp.get_names()} {inp.get_shape()}")

for idx, out in enumerate(model.outputs):  
    print(f"Output {idx}: {out.get_names()} {out.get_shape()}")

try:
    os.makedirs(model_version_dir)
except OSError as ex:
    pass  # ignore existing dir

ov.serialize(model, model_version_dir + "/model.xml", model_version_dir + "/model.bin")

ov_model = ov.Core().read_model(model_version_dir + "/model.xml")
compiled_model = ov.Core().compile_model(model, "CPU")

input_data = np.ones((1, 10),dtype=np.int32)*20
results = compiled_model({"INPUT1": input_data, "INPUT2": input_data})

print(input_data)
print(results)

for idx, out in enumerate(ov_model.outputs):  
   out.get_tensor().set_names({f"out_{idx}"})

ov.save_model(ov_model, "model_with_names.xml")  # it saves model with names