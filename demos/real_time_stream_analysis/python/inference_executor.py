from xml.parsers.expat import model
import ovmsclient
from ovmsclient.tfs_compat.grpc.tensors import NP_TO_TENSOR_MAP
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

class InferenceExecutor:
	def __init__(self, ovms_url, model_name, model_version=0):
		self.ovms_client = ovmsclient.make_grpc_client(ovms_url)
		self.model_name = model_name
		self.model_version = model_version

		model_metadata = self.ovms_client.get_model_metadata(self.model_name)
		if len(model_metadata["inputs"]) > 1 or len(model_metadata["outputs"]) > 1:
			raise ValueError("Unexpected number of model inputs or outputs. Expecting single input and single output")
		
		self.input_name = next(iter(model_metadata['inputs']))
		logger.info("Inference Executor initialized successfully")

	def predict(self, frame):
			frame = np.expand_dims(frame, axis=0)
			input_data = ovmsclient.make_tensor_proto(frame, dtype=NP_TO_TENSOR_MAP[np.float32].TensorDtype)
			result = self.ovms_client.predict({self.input_name: input_data}, self.model_name, self.model_version)
			logger.debug("Got inference results")
			return result

