import os
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from pyovms import Tensor
from model import LogisticRegressionTorch, ModelClass
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_PATH = "/workspace/model/generic_model/model.pt"
ENCODER_PATH = "/workspace/model/generic_model/label_encoder.joblib"
META_PATH = "/workspace/model/generic_model/meta.json"

AVAILABLE_MODEL_CLASSES = {
    "LogisticRegressionTorch": LogisticRegressionTorch
    "KMeansSKLearn" : KMeansSKLearn
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

            model_obj = AVAILABLE_MODEL_CLASSES[model_class_name]()

            if mode == "train":
                if y is None:
                    raise ValueError("y labels are required for training")
                y = np.array(y)
                le = LabelEncoder()
                y_enc = le.fit_transform(y)

                trained = model_obj.fit(X, y_enc, params.get("train_params", {}))

                torch.save(trained.model.state_dict(), MODEL_PATH)
                joblib.dump(le, ENCODER_PATH)
                with open(META_PATH, "w") as f:
                    json.dump({"num_features": X.shape[1], "num_classes": len(le.classes_)}, f)

                preds, _ = trained.predict(X)
                acc = accuracy_score(y_enc, preds)
                prec = precision_score(y_enc, preds, average='weighted', zero_division=0)
                rec = recall_score(y_enc, preds, average='weighted', zero_division=0)
                f1 = f1_score(y_enc, preds, average='weighted', zero_division=0)

                return [Tensor("pipeline_output", np.array([1.0, acc, prec, rec, f1], dtype=np.float32))]

            elif mode == "infer":
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError("No model checkpoint found")

                with open(META_PATH, "r") as f:
                    meta = json.load(f)
                num_features = meta["num_features"]
                num_classes = meta["num_classes"]

                model_obj.model = nn.Linear(num_features, num_classes)
                model_obj.model.load_state_dict(torch.load(MODEL_PATH))
                model_obj.model.eval()

                preds, probs = model_obj.predict(X)
                le = joblib.load(ENCODER_PATH)
                labels = le.inverse_transform(preds)

                response = []
                for label, prob in zip(labels, probs):
                    prob_dict = {le.classes_[i]: float(p) for i, p in enumerate(prob)}
                    response.append({"label": label, "probabilities": prob_dict})

                return [Tensor("pipeline_output", np.array(response, dtype=object))]

            else:
                raise ValueError(f"Unknown mode '{mode}'")

        except Exception as e:
            print(f"[ERROR] {e}")
            return [Tensor("pipeline_output", np.array([f"ERROR: {e}"], dtype=object))]

    def finalize(self):
        print("[finalize] Python node finalized", flush=True)
