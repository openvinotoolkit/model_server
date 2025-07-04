import os
import sys
import pandas as pd
import numpy as np
import tritonclient.grpc as grpcclient
import json

def main():
    DEFAULT_CSV = "workspace/data/iris_train.csv"  
    SERVER_URL = "localhost:9000"
    MODEL_NAME = "pipeline"  

    if len(sys.argv) < 3 or sys.argv[1] not in ("train", "infer"):
        print("Usage: python client_train.py <train|infer> <path_to_csv>")
        sys.exit(1)

    mode = sys.argv[1]
    csv_path = sys.argv[2]

    if not os.path.isfile(csv_path):
        print(f"ERROR: Could not find CSV file: {csv_path}")
        print("Please check your path and try again.")
        sys.exit(1)
     
    try:
        df = pd.read_csv(csv_path)
        print("read CSV file successfully")
    except Exception as e:
        print(f"ERROR: Could not read CSV file: {e}")
        sys.exit(1)

    if mode == "train":
        if "Species" not in df.columns:
            print("ERROR: Training CSV must contain a 'species' column as the label.")
            sys.exit(1)
        print("Training mode detected. Preparing data for training...")

    csv_str = df.to_csv(index=False)

    input_dict = {
        "mode": mode, 
        "data": csv_str
    }

    input_bytes = json.dumps(input_dict).encode("utf-8")

    pipeline_input = np.array([input_bytes], dtype=object)

    inputs = {"pipeline_input": pipeline_input}
    print("prepped payload")

    try:
        client = grpcclient.InferenceServerClient(url=SERVER_URL)
        print(f"Connected to OVMS at {SERVER_URL}")
    except Exception as e:
        print(f"ERROR: Could not connect to OVMS at {SERVER_URL}: {e}")
        sys.exit(1)

    infer_input = grpcclient.InferInput("pipeline_input", pipeline_input.shape, "BYTES")
    infer_input.set_data_from_numpy(pipeline_input)
    print("MODEL_NAME:", MODEL_NAME)
    print("infer_input:", infer_input.name(), infer_input.shape())
    try:
        response = client.infer(
            model_name=MODEL_NAME,
            inputs=[infer_input]
        )
        result = response.as_numpy("pipeline_output")
        print("Server response:", result)
    except Exception as e:
        print(f"ERROR: Inference call failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
