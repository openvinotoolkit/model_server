
import os
import sys
import json
import pandas as pd
import numpy as np
import tritonclient.grpc as grpcclient

def main():
    SERVER_URL = "localhost:9000"
    MODEL_NAME = "pipeline"

    if len(sys.argv) < 4 or sys.argv[1] not in ("train", "infer"):
        print("Usage: python client_train.py <train|infer> <path_to_csv> <target_column> "
              "[--params <path_to_params_json>] [--encode <col1,col2,...>]")
        sys.exit(1)

    mode = sys.argv[1]
    csv_path = sys.argv[2]
    target_column = sys.argv[3]

    params_path = None
    encode_cols = []

    if "--params" in sys.argv:
        idx = sys.argv.index("--params")
        if idx + 1 < len(sys.argv):
            params_path = sys.argv[idx + 1]

    if "--encode" in sys.argv:
        idx = sys.argv.index("--encode")
        if idx + 1 < len(sys.argv):
            encode_cols = sys.argv[idx + 1].split(",")

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
        print(" Training mode detected. Preparing dataset...")

    params = {}
    if params_path:
        try:
            with open(params_path, "r") as f:
                params = json.load(f)
            print(f" Loaded hyperparameters: {params}")
        except Exception as e:
            print(f"ERROR: Could not read params JSON: {e}")
            sys.exit(1)

    csv_str = df.to_csv(index=False)
    input_dict = {
        "mode": mode,
        "data": csv_str,
        "params": params,
        "target_column": target_column,
        "encode_columns": encode_cols
    }

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
            print("Server response:", result[0].decode())
        else:
            print("Server response (raw):", str(result))

    except Exception as e:
        print(f"ERROR: Inference call failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
