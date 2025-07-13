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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

MODEL_PATH = "/workspace/model/iris_logreg/1/model.onnx"
LABEL_COLUMN = "Species"
DROP_COLUMNS = ["Id"]

class OvmsPythonModel:
    def initialize(self, kwargs):
        print("[initialize] Python node initialized", file=sys.stderr, flush=True)

    def execute(self, inputs):
        print("==== [execute] Python node called ====", file=sys.stderr, flush=True)
        try:
            input_tensor = inputs[0]
            input_data = input_tensor.data
            inp_bytes = bytes(input_tensor.data) 

            print("input_data preview:", inp_bytes[:40], file=sys.stderr)
            if inp_bytes[:1] != b'{' and inp_bytes.find(b'{') > 0:
                first_brace = inp_bytes.find(b'{')
                inp_bytes = inp_bytes[first_brace:]
            print("RAW BYTES:", inp_bytes[:100], file=sys.stderr)
            payload = json.loads(inp_bytes.decode("utf-8"))
            mode = payload.get("mode")
            csv_str = payload.get("data")
            if not isinstance(csv_str, str):
                raise ValueError("Missing or invalid 'data' field")

            df = pd.read_csv(StringIO(csv_str))

            if "Id" in df.columns:
                df = df.drop(columns=DROP_COLUMNS)

            if mode == "train":
                if LABEL_COLUMN not in df.columns:
                    raise ValueError(f"Missing label column '{LABEL_COLUMN}' in input data")

                X = df.drop(columns=[LABEL_COLUMN]).values.astype(np.float32)
                y = df[LABEL_COLUMN].values

                model = LogisticRegression(max_iter=200)
                model.fit(X, y)

                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
                onnx_model = convert_sklearn(model, initial_types=initial_type)
                with open(MODEL_PATH, "wb") as f:
                    f.write(onnx_model.SerializeToString())

                y_pred = model.predict(X)
                acc = accuracy_score(y, y_pred)
                prec = precision_score(y, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
                report = classification_report(y, y_pred)
                print(f"[METRICS][train] accuracy={acc}, precision={prec}, recall={rec}, f1={f1}", file=sys.stderr)
                print(f"[METRICS][train] classification_report:\n{report}", file=sys.stderr, flush=True)

                metrics_str = f"acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}"
                result = np.array([1.0, acc, prec, rec, f1], dtype=np.float32)
                return [Tensor("pipeline_output", result)]

            elif mode == "infer":
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError("Model not trained yet")

                y_true = None
                if LABEL_COLUMN in df.columns:
                    y_true = df[LABEL_COLUMN].values
                    df = df.drop(columns=[LABEL_COLUMN])

                X = df.values.astype(np.float32)
                sess = ort.InferenceSession(MODEL_PATH)
                input_name = sess.get_inputs()[0].name
                preds = sess.run(None, {input_name: X})[0]

                if preds.ndim > 1 and preds.shape[1] == 1:
                    preds = preds.ravel()

                label_map = {
                    "Iris-setosa": 0,
                    "Iris-versicolor": 1,
                    "Iris-virginica": 2
                }
                if isinstance(preds, str):
                    preds = np.array([preds])
                if preds.dtype.type is np.str_ or preds.dtype.type is np.object_:
                    res_int = np.vectorize(label_map.get)(preds)
                else:
                    res_int = preds

                result = res_int.astype(np.float32)

                if y_true is not None:
                    if y_true.dtype.type is np.str_ or y_true.dtype.type is np.object_:
                        y_true_mapped = np.vectorize(label_map.get)(y_true)
                    else:
                        y_true_mapped = y_true

                    acc = accuracy_score(y_true_mapped, res_int)
                    prec = precision_score(y_true_mapped, res_int, average='weighted', zero_division=0)
                    rec = recall_score(y_true_mapped, res_int, average='weighted', zero_division=0)
                    f1 = f1_score(y_true_mapped, res_int, average='weighted', zero_division=0)
                    report = classification_report(y_true_mapped, res_int)
                    print(f"[METRICS][infer] accuracy={acc}, precision={prec}, recall={rec}, f1={f1}", file=sys.stderr)
                    print(f"[METRICS][infer] classification_report:\n{report}", file=sys.stderr, flush=True)

                return [Tensor("pipeline_output", result)]
            else:
                raise ValueError(f"Unknown mode '{mode}'")
        except Exception as e:
            print(f"[ERROR] Exception in execute: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            error_msg = f"ERROR: {str(e)}".encode()
            return [Tensor("pipeline_output", np.array([error_msg], dtype=object))]

    def finalize(self):
        print("[finalize] Python node finalized", file=sys.stderr, flush=True)
