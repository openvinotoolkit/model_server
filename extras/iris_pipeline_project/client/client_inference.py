import numpy as np
import tritonclient.grpc as grpcclient
import pandas as pd
import json
import sys
import os

SERVER_URL = "localhost:9000"
MODEL_NAME = "pipeline"

def print_usage():
    print("Usage: python client_inference.py infer <path_to_csv> [--target_column <column>] [--labelmap <path_to_labelmap_json>]")
    sys.exit(1)

def main():
    if len(sys.argv) < 3 or sys.argv[1] != "infer":
        print_usage()

    mode = sys.argv[1]
    csv_path = sys.argv[2]
    target_column = None
    labelmap_path = None

    for i in range(3, len(sys.argv)):
        if sys.argv[i] == "--target_column" and i+1 < len(sys.argv):
            target_column = sys.argv[i+1]
        elif sys.argv[i] == "--labelmap" and i+1 < len(sys.argv):
            labelmap_path = sys.argv[i+1]

    if not os.path.isfile(csv_path):
        print(f"ERROR: Could not find CSV file: {csv_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
        print("CSV loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}")
        sys.exit(1)

    if target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])

    csv_str = df.to_csv(index=False)
    input_dict = {
        "mode": mode,
        "data": csv_str,
        "target_column": target_column
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

        if isinstance(result, np.ndarray) and result.dtype == np.float32:
            print(" Server responded with float32 array.")
            # format: [1.0, label, confidence]
            if len(result) >= 3:
                status = result[0]
                label = int(result[1])
                confidence = float(result[2])
                print(f" Inference Result:")
                print(f"Prediction: {label} (Confidence: {confidence:.2f})")
            else:
                print(" Unexpected result shape:", result)
        else:
            print(" Server response (raw):", str(result))

    except Exception as e:
        print(f"ERROR: Inference call failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
