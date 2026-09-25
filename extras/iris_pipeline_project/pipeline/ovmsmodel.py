import os
import json
import numpy as np
import joblib
from pyovms import Tensor
from model import LogisticRegressionTorch, KMeansSkLearn  # add more in future

MODEL_PATH = "/workspace/model/generic_model/model.bin"
META_PATH = "/workspace/model/generic_model/meta.json"

AVAILABLE_MODEL_CLASSES = {
    "LogisticRegressionTorch": LogisticRegressionTorch,
    "KMeansSkLearn": KMeansSkLearn,
}

class OvmsPythonModel:
    def initialize(self, kwargs):
        print("[initialize] Python node initialized", flush=True)
        self.model_obj = None

    def execute(self, inputs):
        try:
            inp_bytes = bytes(inputs[0].data)
            first_brace = inp_bytes.find(b'{')
            if first_brace > 0:
                inp_bytes = inp_bytes[first_brace:]
            payload = json.loads(inp_bytes.decode("utf-8"))

            mode = payload.get("mode")
            X = np.array(payload.get("X"), dtype=np.float32)
            y = payload.get("y", None)
            params = payload.get("params", {})
            model_class_name = payload.get("model_class", "LogisticRegressionTorch")

            if model_class_name not in AVAILABLE_MODEL_CLASSES:
                raise ValueError(f"Unknown model: {model_class_name}")

            ModelClass = AVAILABLE_MODEL_CLASSES[model_class_name]
            model_obj = ModelClass()

            if mode == "train":
                trained = model_obj.fit(X, y, params.get("train_params", {}))
                model_obj.save(MODEL_PATH, META_PATH)
                
                metrics = {}
                if hasattr(model_obj, "evaluate") and y is not None:
                    metrics = model_obj.evaluate(X, y)

                response = {"status": "trained", "metrics": metrics}
                json_bytes = json.dumps(response).encode("utf-8")
                return [Tensor("pipeline_output", np.array([json_bytes], dtype=object))]

            elif mode == "infer":
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError("No model checkpoint found")

                model_obj.load(MODEL_PATH, META_PATH)
                predictions = model_obj.predict(X)

                response = {"predictions": predictions}
                json_bytes = json.dumps(response).encode("utf-8")
                return [Tensor("pipeline_output", np.array([json_bytes], dtype=object))]

            else:
                raise ValueError(f"Unknown mode '{mode}'")

        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
            return [Tensor("pipeline_output", np.array([f"ERROR: {e}"], dtype=object))]

    def finalize(self):
        print("[finalize] Python node finalized", flush=True)
