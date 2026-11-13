import pandas as pd
import numpy as np
import joblib
import os
import io
from pyovms import Tensor
from sklearn.linear_model import LogisticRegression
from skl2onnx import to_onnx
import onnx

class OvmsPythonModel:
    def initialize(self, kwargs):
        print("Training handler initialized.")

    def execute(self, inputs):
        print("Starting training...")
        input_tensor = inputs[0]
        input_data = input_tensor.as_numpy()
        csv_str = input_data.tobytes().decode('utf-8')

        df = pd.read_csv(io.StringIO(csv_str))

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        model = LogisticRegression(max_iter=200)
        model.fit(X, y)

        # Ensure output directory exists
        output_dir = "/model/iris_logreg/1"
        os.makedirs(output_dir, exist_ok=True)

        # Convert to ONNX and save to the versioned folder
        onx = to_onnx(model, X[:1], target_opset=12)
        output_path = os.path.join(output_dir, "logreg_model.onnx")
        with open(output_path, "wb") as f:
            f.write(onx.SerializeToString())

        print(f"Model saved to {output_path}")
        return [Tensor.from_numpy(np.array([True]))]

    def finalize(self):
        print("Training handler finalized.")