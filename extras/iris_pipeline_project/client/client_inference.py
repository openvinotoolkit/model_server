import numpy as np
import tritonclient.grpc as grpcclient
import pandas as pd
import json
import sys
import os

df = pd.read_csv("data/iris_test_stripped.csv")
if "Species" in df.columns:
    df = df.drop(columns=["Species"])  
csv_str = df.to_csv(index=False)
input_dict = {"mode": "infer", "data": csv_str}
input_bytes = json.dumps(input_dict).encode("utf-8")
pipeline_input = np.array([input_bytes], dtype=object)

if len(sys.argv) < 3 or sys.argv[1] not in ("train", "infer"):
    print("Usage: python client_inference.py <infer> <path_to_csv>")
    sys.exit(1)

mode = sys.argv[1]
csv_path = sys.argv[2]

if not os.path.isfile(csv_path):
    print(f"ERROR: Could not find CSV file: {csv_path}")
    sys.exit(1)

try:
    df = pd.read_csv(csv_path)
    print("Read CSV file successfully")
except Exception as e:
    print(f"ERROR: Could not read CSV file: {e}")
    sys.exit(1)
input_name = "pipeline_input"
infer_input = grpcclient.InferInput(input_name, pipeline_input.shape, "BYTES")
infer_input.set_data_from_numpy(pipeline_input)

if mode == "train":
    if "Species" not in df.columns:
        print("ERROR: Training CSV must contain a 'Species' column as the label.")
        sys.exit(1)
    print("Training mode detected. Preparing data for training...")

csv_str = df.to_csv(index=False)
if "Species" in df.columns:
    df = df.drop(columns=["Species"])  
input_dict = {"mode": mode, "data": csv_str}
input_bytes = json.dumps(input_dict).encode("utf-8")
pipeline_input = np.array([input_bytes], dtype=object)

input_name = "pipeline_input"
infer_input = grpcclient.InferInput(input_name, pipeline_input.shape, "BYTES")
infer_input.set_data_from_numpy(pipeline_input)

print("Inference mode detected.")
client = grpcclient.InferenceServerClient(url="localhost:9000")
model_name = "pipeline"

response = client.infer(model_name, [infer_input])
preds = response.as_numpy("pipeline_output")

if preds == 0.0:
    result_string = "Iris-setosa"
elif preds == 1.0:
    result_string =  "Iris-versicolor"
else:
    result_string = "Iris-virginica"

print("Inference predictions:", result_string)
