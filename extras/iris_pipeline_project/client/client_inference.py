import numpy as np
import tritonclient.grpc as grpcclient
import pandas as pd
import json

df = pd.read_csv("data/iris_test_stripped.csv")
if "Species" in df.columns:
    df = df.drop(columns=["Species"])  
csv_str = df.to_csv(index=False)
input_dict = {"mode": "infer", "data": csv_str}
input_bytes = json.dumps(input_dict).encode("utf-8")
pipeline_input = np.array([input_bytes], dtype=object)

input_name = "pipeline_input"
infer_input = grpcclient.InferInput(input_name, pipeline_input.shape, "BYTES")
infer_input.set_data_from_numpy(pipeline_input)

client = grpcclient.InferenceServerClient(url="localhost:9000")
model_name = "pipeline"

response = client.infer(model_name, [infer_input])
preds = response.as_numpy("pipeline_output")

print("Inference predictions:", preds)
