import queue
from xml.parsers.expat import model
import ovmsclient
from ovmsclient.tfs_compat.grpc.tensors import NP_TO_TENSOR_MAP
import numpy as np
from logger import get_logger
import threading

class ExecutorThread(threading.Thread):
	def __init__(self, index, inference_executor):
		print(f"Initializing requesting thread index: {index}")
		super().__init__()
		self.index = index
		self.inference_executor = inference_executor
		self.input = None
		self.result = None
		self.predict_durations = []
		self.input_ready_event = threading.Event()
		self.result_ready_event = threading.Event()

	def is_initialized(self):
		return not (self.input is None and self.result is None)

	def wait_for_input(self):
		self.input_ready_event.wait()
		self.input_ready_event.clear()

	def wait_for_result(self):
		self.result_ready_event.wait()
		self.result_ready_event.clear()

	def notify_input_ready(self):
		self.input_ready_event.set()

	def notify_result_ready(self):
		self.result_ready_event.set()

	def set_input(self, frame):
		self.input = frame
		self.notify_input_ready()

	def get_result(self):
		return self.result

	def run(self):
		print(f"Launching requesting thread index: {self.index}")
		global force_exit
		while (True):
			self.wait_for_input()
			self.result = self.inference_executor.predict(self.input)
			self.notify_result_ready()
		print(f"Stopping requesting thread index: {self.index}")



class InferenceExecutor:
	def __init__(self, ovms_url, model_name, model_version=0, num_threads=32, 
				 inputs_queue_maxsize=1000, results_queue_maxsize=1000):
		
		self.logger = get_logger(__name__)

		self.ovms_client = ovmsclient.make_grpc_client(ovms_url)
		self.model_name = model_name
		self.model_version = model_version

		self.inputs_queue = queue.Queue(maxsize=inputs_queue_maxsize)
		self.results_queue = queue.Queue(maxsize=results_queue_maxsize)

		model_metadata = self.ovms_client.get_model_metadata(self.model_name)
		if len(model_metadata["inputs"]) > 1 or len(model_metadata["outputs"]) > 1:
			raise ValueError("Unexpected number of model inputs or outputs. Expecting single input and single output")
		
		self.input_name = next(iter(model_metadata['inputs']))

		self.executor_threads = [ExecutorThread(i, self) for i in range(num_threads)]
		for executor_thread in self.executor_threads:
			executor_thread.start()

		self.main_thread = threading.Thread(target=self._main_thread)
		self.main_thread.start()

		self.logger.info("Inference Executor initialized successfully")

	def predict(self, frame):
			frame = np.expand_dims(frame, axis=0)
			input_data = ovmsclient.make_tensor_proto(frame, dtype=NP_TO_TENSOR_MAP[np.float32].TensorDtype)
			result = self.ovms_client.predict({self.input_name: input_data}, self.model_name, self.model_version)
			return result

	def schedule_inference(self, frame) -> bool:
		try:
			self.inputs_queue.put_nowait(frame)
			return True
		except queue.Full:
			return False

	def pull_result(self):
		try:
			return self.results_queue.get_nowait()
		except queue.Empty:
			return None, None

	def _main_thread(self):
		num_threads = len(self.executor_threads)
		i = 0
		while True:
			input = self.inputs_queue.get()
			if not self.executor_threads[i].is_initialized():
				self.executor_threads[i].set_input(input)
				i = (i + 1) % num_threads
				continue
			self.executor_threads[i].wait_for_result()
			inferred_frame = self.executor_threads[i].input
			result = self.executor_threads[i].get_result()
			self.results_queue.put((inferred_frame, result))
			self.executor_threads[i].set_input(input)
			i = (i + 1) % num_threads



