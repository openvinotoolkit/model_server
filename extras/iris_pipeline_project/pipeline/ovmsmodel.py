import numpy as np
import pandas as pd
import os
import json
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
import onnxruntime as ort

MODEL_PATH = "/workspace/model/iris_logreg/1/model.onnx"
LABEL_COLUMN = "species"

class OvmsPythonModel:
    def initialize(self, kwargs):
        print("Training handler initialized.")

    def execute(self, inputs, outputs, parameters, context):
        # Expecting a dict: {"mode": "train" or "infer", "data": <CSV string>}
        input_bytes = inputs["pipeline_input"]
        try:
            input_str = input_bytes.tobytes().decode('utf-8')
            input_obj = json.loads(input_str)
            mode = input_obj.get("mode")
            csv_str = input_obj.get("data")
        except Exception as e:
            outputs["pipeline_output"] = np.array([f"ERROR: Invalid input format: {e}"], dtype=object)
            return

        try:
            df = pd.read_csv(pd.compat.StringIO(csv_str))
        except Exception as e:
            outputs["pipeline_output"] = np.array([f"ERROR: Could not parse CSV: {e}"], dtype=object)
            return

        if mode == "train":
            if LABEL_COLUMN not in df.columns:
                outputs["pipeline_output"] = np.array([f"ERROR: Training data must include label column '{LABEL_COLUMN}'"], dtype=object)
                return
            X = df.drop(columns=[LABEL_COLUMN])
            y = df[LABEL_COLUMN]
            model = LogisticRegression(max_iter=200)
            model.fit(X, y)
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            with open(MODEL_PATH, "wb") as f:
                f.write(onnx_model.SerializeToString())
            outputs["pipeline_output"] = np.array(["training complete"], dtype=object)
        elif mode == "infer":
            if not os.path.exists(MODEL_PATH):
                outputs["pipeline_output"] = np.array(["ERROR: Model not trained yet"], dtype=object)
                return
            X = df.values.astype(np.float32)
            sess = ort.InferenceSession(MODEL_PATH)
            input_name = sess.get_inputs()[0].name
            preds = sess.run(None, {input_name: X})[0]
            outputs["pipeline_output"] = preds
        else:
            outputs["pipeline_output"] = np.array([f"ERROR: Unknown mode '{mode}'"], dtype=object)

    def finalize(self):
        print("Training handler finalized.")
