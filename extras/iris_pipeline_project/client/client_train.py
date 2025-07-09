import os
import sys
import pandas as pd
import numpy as np
import tritonclient.grpc as grpcclient
import json

def main():
    SERVER_URL = "localhost:9000"
    MODEL_NAME = "pipeline"

    if len(sys.argv) < 3 or sys.argv[1] not in ("train", "infer"):
        print("Usage: python client_train.py <train|infer> <path_to_csv>")
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

    if mode == "train":
        if "Species" not in df.columns:
            print("ERROR: Training CSV must contain a 'Species' column as the label.")
            sys.exit(1)
        print("Training mode detected. Preparing data for training...")

    csv_str = df.to_csv(index=False)
    input_dict = {"mode": mode, "data": csv_str}
    input_bytes = json.dumps(input_dict).encode()

    pipeline_input = np.array([input_bytes], dtype=object)

    try:
        client = grpcclient.InferenceServerClient(url=SERVER_URL)
        print(f"Connected to OVMS at {SERVER_URL}")
    except Exception as e:
        print(f"ERROR: Could not connect to OVMS at {SERVER_URL}: {e}")
        sys.exit(1)

    infer_input = grpcclient.InferInput("pipeline_input", pipeline_input.shape, "BYTES")
    infer_input.set_data_from_numpy(pipeline_input)

    try:
        response = client.infer(
            model_name=MODEL_NAME,
            inputs=[infer_input]
        )
        result = response.as_numpy("pipeline_output")
        
        if isinstance(result, np.ndarray) and result.dtype == object:
            print("Server response decoded obj:", result[0].decode())
        elif isinstance(result, np.ndarray) and result.dtype == np.uint8:
            print("Server response decoded int8:", bytes(result).decode())
        elif isinstance(result, (bytes, bytearray)):
            print("Server response decoded bytarray:", result.decode())
        else:
            print("Server response decoded: string - ", str(result))
            print("The output string formatted as: [<1 - Model trained successfully | 0 - Otherwise>   <Accuracy>  <Precision>  <Recall>  <f1-score>]")
    except Exception as e:
        print(f"ERROR: Inference call failed or output decoding failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()