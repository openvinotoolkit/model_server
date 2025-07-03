import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
from pyovms import Tensor

from io import StringIO  # Use this for reading CSV from string

MODEL_PATH = "/workspace/model/iris_logreg/1/model.onnx"
LABEL_COLUMN = "Species"

class OvmsPythonModel:
    def initialize(self, kwargs):
        print("[initialize] Python node initialized", file=sys.stderr, flush=True)

    def execute(self, inputs):
        print("==== [execute] Python node called ====", file=sys.stderr, flush=True)

        try:
            input_tensor = inputs[0]  # pyovms.Tensor
            input_data = input_tensor.data  # Raw bytes
            payload = json.loads(input_data.decode("utf-8"))

            mode = payload.get("mode")
            csv_str = payload.get("data")
            if not isinstance(csv_str, str):
                raise ValueError("Missing or invalid 'data' field")

            df = pd.read_csv(StringIO(csv_str))

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

                result = "training complete".encode("utf-8")
                return [Tensor("pipeline_output", result)]

            elif mode == "infer":
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError("Model not trained yet")

                X = df.values.astype(np.float32)
                sess = ort.InferenceSession(MODEL_PATH)
                input_name = sess.get_inputs()[0].name
                preds = sess.run(None, {input_name: X})[0]

                result = json.dumps(preds.tolist()).encode("utf-8")
                return [Tensor("pipeline_output", result)]

            else:
                raise ValueError(f"Unknown mode '{mode}'")

        except Exception as e:
            print(f"[ERROR] Exception in execute: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            error_msg = f"ERROR: {str(e)}".encode("utf-8")
            return [Tensor("pipeline_output", error_msg)]

    def finalize(self):
        print("[finalize] Python node finalized", file=sys.stderr, flush=True)
