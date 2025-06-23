from pyovms import Tensor
import numpy as np
from scipy.special import softmax
from tritonclient.utils import deserialize_bytes_tensor

class OvmsPythonModel:

    def initialize(self, kwargs:dict):
        pass

    def execute(self, inputs:list):
        embedding=inputs[0].as_numpy()
        norm=np.linalg.norm(embedding, axis=1, keepdims=True)
        normalized_embedding=embedding/norm

        return[Tensor(name="embedding", data=normalized_embedding.astype(np.float32))]
