
import os
import sys
import json
import pandas as pd
import numpy as np
import tritonclient.grpc as grpcclient
from sklearn.preprocessing import LabelEncoder


def main():
    SERVER_URL = "localhost:9000"
    MODEL_NAME = "pipeline"

    if len(sys.argv) < 4 or sys.argv[1] not in ("train", "infer"):
        print("Usage: python client_train.py <train|infer> <path_to_csv> <target_column (or NONE for KMeans)> "
              "[--params <path_to_params_json>] [--encode <col1,col2,...>] [--model_class <ModelClassName>]")
        sys.exit(1)

    mode = sys.argv[1]
    csv_path = sys.argv[2]
    target_column = sys.argv[3] if sys.argv[3] != "NONE" else None

    params_path = None
    encode_cols = []
    model_class_name = "LogisticRegressionTorch"

    if "--params" in sys.argv:
        idx = sys.argv.index("--params")
        if idx + 1 < len(sys.argv):
            params_path = sys.argv[idx + 1]

    if "--encode" in sys.argv:
        idx = sys.argv.index("--encode")
        if idx + 1 < len(sys.argv):
            encode_cols = sys.argv[idx + 1].split(",")

    if "--model_class" in sys.argv:
        idx = sys.argv.index("--model_class")
        if idx + 1 < len(sys.argv):
            model_class_name = sys.argv[idx + 1]

    if not os.path.isfile(csv_path):
        print(f"ERROR: Could not find CSV file: {csv_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
        print("Read CSV file successfully")
    except Exception as e:
        print(f"ERROR: Could not read CSV file: {e}")
        sys.exit(1)

    if encode_cols:
        for col in encode_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            else:
                print(f"WARNING: Encode column '{col}' not found in CSV")

    if model_class_name == "KMeans":
        X = df.values
        y = None
    else:
        if not target_column or target_column not in df.columns:
            print(f"ERROR: Target column '{target_column}' not found in CSV")
            sys.exit(1)
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values.tolist() if mode == "train" else None

    params = {}
    if params_path:
        try:
            with open(params_path, "r") as f:
                params = json.load(f)
            print(f"Loaded hyperparameters: {params}")
        except Exception as e:
            print(f"ERROR: Could not read params JSON: {e}")
            sys.exit(1)

    payload = {
        "mode": mode,
        "X": X.tolist(),
        "y": y,
        "params": params,
        "model_class": model_class_name
    }

    input_bytes = json.dumps(payload).encode("utf-8")
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
            print("Server response:")
            for item in result:
                if isinstance(item, (bytes, bytearray)):
                    try:
                        item = item.decode()
                    except Exception:
                        pass
                try:
                    response_data = json.loads(item)
                    print(json.dumps(response_data, indent=2))
                except Exception:
                    print(item)
        else:
            print("Model trained successfully")

    except Exception as e:
        print(f"ERROR: Inference call failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()