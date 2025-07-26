import os
import sys
import json
import numpy as np
import pandas as pd
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
from pyovms import Tensor
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

MODEL_PATH = "/workspace/model/generic_model/1/model.onnx"
ENCODER_PATH = "/workspace/model/generic_model/1/label_encoder.joblib"
META_PATH = "/workspace/model/generic_model/1/meta.json"

class OvmsPythonModel:
    def initialize(self, kwargs):
        print("[initialize] Python node initialized", file=sys.stderr, flush=True)

    def execute(self, inputs):
        print("==== [execute] Python node called ====", file=sys.stderr, flush=True)
        try:
            input_tensor = inputs[0]
            inp_bytes = bytes(input_tensor.data)
            first_brace = inp_bytes.find(b'{')
            if first_brace > 0:
                inp_bytes = inp_bytes[first_brace:]

            payload = json.loads(inp_bytes.decode("utf-8"))
            mode = payload.get("mode")
            csv_str = payload.get("data")
            target_column = payload.get("target_column")  
            print("Received payload:", csv_str)

            if not isinstance(csv_str, str):
                raise ValueError("Missing or invalid 'data' field")

            df = pd.read_csv(StringIO(csv_str))

            if mode == "train":
                if target_column not in df.columns:
                    raise ValueError(f"Label column '{target_column}' not found")

                X_df = df.drop(columns=[target_column])
                y = df[target_column].values
                feature_names = list(X_df.columns)

                print("Handling missing values...")
                imp = SimpleImputer(strategy='mean')
                X = imp.fit_transform(X_df).astype(np.float32)

                le = LabelEncoder()
                y_enc = le.fit_transform(y)
                class_names = list(le.classes_)

                os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)
                joblib.dump(le, ENCODER_PATH)
                meta = {
                    "target_column": target_column,
                    "feature_names": feature_names,
                    "class_names": class_names
                }
                with open(META_PATH, "w") as f:
                    json.dump(meta, f)

                model = LogisticRegression(**payload.get("params", {}))
                model.fit(X, y_enc)
                print("Intercept:", model.intercept_)
                print("Coefficients:", model.coef_)

                initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
                onnx_model = convert_sklearn(model, initial_types=initial_type)
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                with open(MODEL_PATH, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                print("[DEBUG] Training complete. Saved model to:", MODEL_PATH)

                y_pred = model.predict(X)
                acc = accuracy_score(y_enc, y_pred)
                prec = precision_score(y_enc, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_enc, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_enc, y_pred, average='weighted', zero_division=0)

                print(f"[TRAIN METRICS] acc={acc}, prec={prec}, rec={rec}, f1={f1}", file=sys.stderr)
                result = np.array([1.0, acc, prec, rec, f1], dtype=np.float32)
                return [Tensor("pipeline_output", result)]
            elif mode == "infer":
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError("Trained model not found")
                if not os.path.exists(META_PATH):
                    raise FileNotFoundError("Metadata file missing")

                with open(META_PATH, "r") as f:
                    meta = json.load(f)
                feature_names = meta["feature_names"]
                class_names = meta.get("class_names", [])

                if any(f not in df.columns for f in feature_names):
                    missing = [f for f in feature_names if f not in df.columns]
                    raise ValueError(f"Missing required feature(s): {missing}")

                X_df = df[feature_names]
                imp = SimpleImputer(strategy='mean')
                X = imp.fit_transform(X_df).astype(np.float32)

                sess = ort.InferenceSession(MODEL_PATH)
                input_name = sess.get_inputs()[0].name
                output_names = [output.name for output in sess.get_outputs()]
                outputs = sess.run(output_names, {input_name: X})
                print("[DEBUG] Model inference outputs:", output_names)
                print("outputs: ", outputs)

                if len(outputs) == 2:
                    label_indices, probs = outputs
                    result_array = []

                    for i in range(len(label_indices)):
                        pred_index = int(label_indices[i])

                        prob_dict = probs[i]

                        confidence = float(prob_dict[pred_index])

                        result_array.append([1.0, pred_index, confidence])

                        label_name = class_names[pred_index] if class_names else str(pred_index)
                        print(f"Sample {i} => Predicted Label: {label_name} (index: {pred_index}), Confidence: {confidence}", file=sys.stderr)

                    return [Tensor("pipeline_output", np.array(result_array, dtype=np.float32))]


                raise RuntimeError("Unexpected ONNX model output structure.")

            else:
                raise ValueError(f"Unknown mode '{mode}'")

        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            err = f"ERROR: {str(e)}".encode()
            return [Tensor("pipeline_output", np.array([err], dtype=object))]

    def finalize(self):
        print("[finalize] Python node finalized", file=sys.stderr, flush=True)
