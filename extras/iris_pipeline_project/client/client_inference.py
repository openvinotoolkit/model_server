import numpy as np
import tritonclient.grpc as grpcclient
import pandas as pd
import json
import sys
import os
import matplotlib.pyplot as plt
SERVER_URL = "localhost:9000"
MODEL_NAME = "pipeline"

def print_usage():
    print("Usage: python client_inference.py infer <path_to_csv> [--target_column <column>] [--encode <col1,col2,...>] [--model_class <ModelClassName>]")
    sys.exit(1)

def main():
    if len(sys.argv) < 3 or sys.argv[1] != "infer":
        print_usage()

    mode = sys.argv[1]
    csv_path = sys.argv[2]
    target_column = None
    encode_cols = []
    model_class_name = "LogisticRegressionTorch"

    for i in range(3, len(sys.argv)):
        if sys.argv[i] == "--target_column" and i+1 < len(sys.argv):
            target_column = sys.argv[i+1]
        elif sys.argv[i] == "--encode" and i+1 < len(sys.argv):
            encode_cols = sys.argv[i+1].split(",")
        elif sys.argv[i] == "--model_class" and i+1 < len(sys.argv):
            model_class_name = sys.argv[i+1]

    if "KMeans" in model_class_name or (target_column and target_column.lower() == "none"):
        target_column = None

    if not os.path.isfile(csv_path):
        print(f"ERROR: Could not find CSV file: {csv_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
        print("CSV loaded successfully. kmeans")
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}")
        sys.exit(1)

    if "KMeans" not in model_class_name and target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])

    if encode_cols:
        for col in encode_cols:
            if col in df.columns:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            else:
                print(f"WARNING: Encode column '{col}' not found in CSV")

    X = df.values

    payload = {
        "mode": mode,
        "X": X.tolist(),
        "y": None,
        "params": {},
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

        if result.dtype == object:  
            for item in result:
                if isinstance(item, (bytes, bytearray)):
                    item = item.decode("utf-8")   
                parsed = json.loads(item)
                print("Cluster assignments:", parsed["labels"])
                print("Cluster centroids:", parsed["centroids"])
        else:
            pass

        if result.dtype in [np.float64, np.float32]:
            raw_bytes = result.view(np.uint8).tobytes()
            decoded = raw_bytes.decode("utf-8", errors="ignore").strip("\x00")
            decoded_items = [decoded]


        print("DEBUG result type:", type(result))
        if isinstance(result, np.ndarray):
            print("DEBUG result dtype:", result.dtype)
            print("DEBUG result shape:", result.shape)

        if "KMeans" in model_class_name:
            decoded_items = []
            if isinstance(result, np.ndarray):
                if result.dtype == object:
                    decoded_items = result
                elif result.dtype in [np.float64, np.float32]:
                    raw_bytes = result.tobytes()
                    try:
                        decoded = raw_bytes.decode("utf-8").strip("\x00")
                        decoded_items = [decoded]
                    except Exception as e:
                        pass

            for item in decoded_items:
                if isinstance(item, (bytes, bytearray)):
                    item = item.decode()
                try:
                    response_data = json.loads(item)
                except Exception:
                    response_data = item

                print("KMeans clustering result:")
                if isinstance(response_data, dict) and "labels" in response_data and "centroids" in response_data:
                    labels = response_data["labels"]
                    centroids = np.array(response_data["centroids"])

                    print("Cluster assignments:", labels[:20], "...")  
                    print("Cluster centroids:")
                    for i, centroid in enumerate(centroids):
                        print(f"  Centroid {i}: {centroid}")

                    X = df.drop(columns=[target_column]) if target_column else df
                    X = X.values
                    plt.figure(figsize=(8, 6))
                    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30, alpha=0.7, label="Points")
                    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=200, marker="X", label="Centroids")
                    plt.title("KMeans Clustering Result")
                    plt.xlabel("Feature 1")
                    plt.ylabel("Feature 2")
                    plt.legend()
                    plt.show()
                else:
                    print("Unexpected response:", response_data)

        elif isinstance(result, np.ndarray) and result.dtype == object:
            print("Server responded with object array.")
            for item in result:
                if isinstance(item, (bytes, bytearray)):
                    try:
                        item = item.decode()
                    except Exception:
                        pass
                try:
                    response_data = json.loads(item)
                except Exception:
                    response_data = item
                if isinstance(response_data, list):
                    for entry in response_data:
                        label = entry.get("label")
                        probs = entry.get("probabilities", {})
                        print(f"Prediction: {label}")
                        print("Probabilities:")
                        for k, v in probs.items():
                            print(f"  {k}: {v:.4f}")
                        print("-" * 30)
                elif isinstance(response_data, dict):
                    label = response_data.get("label")
                    probs = response_data.get("probabilities", {})
                    print(f"Prediction: {label}")
                    print("Probabilities:")
                    for k, v in probs.items():
                        print(f"  {k}: {v:.4f}")
                    print("-" * 30)
                else:
                    print(response_data)
        elif isinstance(result, (bytes, bytearray)):
            try:
                decoded = result.decode()
                print("Decoded result:", decoded)
                try:
                    response_data = json.loads(decoded)
                    print(json.dumps(response_data, indent=2))
                except Exception:
                    print(decoded)
            except Exception:
                print("Raw bytes result:", result)
        elif isinstance(result, np.ndarray) and result.dtype in [np.float32, np.float64]:
            print("Server responded with numeric array.")
            print("Values:", result)
        else:
            print("Server response (raw):", str(result))
            

    except Exception as e:
        print(f"ERROR: Inference call failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()