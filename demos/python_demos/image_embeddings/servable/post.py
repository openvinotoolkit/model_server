from pyovms import Tensor
import numpy as np

class OvmsPythonModel:
    def initialize(self, kwargs: dict):
        self.node_name = kwargs.get("node_name", "")
        if "clip" in self.node_name.lower():
            self.mode = "clip"
        elif "dino" in self.node_name.lower():
            self.mode = "dino"
        elif "laion" in self.node_name.lower():
            self.mode = "laion"
        else:
            raise ValueError(f"Unsupported model type in node name: {self.node_name}")

    def execute(self, inputs: list) -> list:
        try:
            tensor = inputs[0]
            embedding = np.frombuffer(tensor.data, dtype=np.float32).reshape(tensor.shape)
            norm = np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-10
            normalized = embedding / norm
            return [Tensor(name="embedding", buffer=normalized.astype(np.float32))]
        except Exception as e:
            print(">>> ERROR in Postprocessor:", str(e))
            raise

