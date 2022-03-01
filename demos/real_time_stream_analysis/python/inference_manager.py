#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import queue
from typing import Tuple
import numpy as np
import threading
import time
import multiprocessing
from logger import get_logger
from inference_executor import InferenceExecutor
from inference_pipeline_handler import InferencePipelineHandler


class InferenceManager:

	def __init__(self, ovms_url, model_name, model_version, num_inference_executors, binary_input,
				 buffer_size):
		
		self.exit_event = threading.Event()
		self.abort_event = threading.Event()

		self.logger = get_logger(__name__)

		model_version_str = "latest" if model_version == 0 else model_version
		self.logger.info(f"OVMS Endpoint spec - ovms_url: {ovms_url}; model_name: {model_name}; model_version: {model_version_str}")

		ovms_info = {
			"ovms_url": ovms_url, 
			"model_name": model_name,
			"model_version": model_version
		}

		self.logger.info(f"Input buffer capacity set to: {buffer_size} frames")
		self.inputs_queue = queue.Queue(maxsize=buffer_size)
		self.results_queue = queue.Queue(maxsize=buffer_size)

		if binary_input:
			self.logger.info("Using binary input switched on")
		
		self.logger.info(f"Number of Inference Executors: {num_inference_executors}")
		self.inference_executors = [InferenceExecutor(i, ovms_info, binary_input,
								   input_queue=multiprocessing.Queue(buffer_size),
								   result_queue=multiprocessing.Queue(buffer_size))
								   for i in range(num_inference_executors)]

	def initialize(self) -> None:
		self.logger.info("Initializing Inference Manager...")
		self.logger.info("Starting Inference Executors...")
		for inference_executor in self.inference_executors:
			inference_executor.start()

		self.logger.info("Starting inference pipeline thread")
		self.inference_pipeline_thread = threading.Thread(target=self._inference_pipeline_thread)
		self.inference_pipeline_thread.start()

		self.logger.info("Starting inference executors monitoring thread")
		self.inference_executors_monitoring_thread = threading.Thread(target=self._inference_executors_monitoring_thread)
		self.inference_executors_monitoring_thread.start()
		self.logger.info("Inference Manager initialized successfully")


	def shutdown(self) -> None:
		self.logger.info("Shutting down Inference Manager...")
		self.logger.info("Exiting Inference Manager thread...")
		self.exit_event.set()

		self.inference_pipeline_thread.join()
		self.logger.info("Inference pipeline thread exited successfully")
		self.inference_executors_monitoring_thread.join()
		self.logger.info("Inference executors monitoring thread exited successfully")

		self.logger.info("Shutting down inference executors...")
		for inference_executor in self.inference_executors:
			inference_executor.shutdown()
			inference_executor.join()
			self.logger.info(f"Inference-Executor-{inference_executor.id} shut down successfully")
		self.logger.info("Inference Executors shut down successfully")
		self.logger.info("Inference Manager shut down successfully")


	def schedule_inference(self, frame) -> bool:
		# Non blocking inference scheduling method. Returns True on success.
		# Returns False if buffer is full and new data cannot be scheduled for inference at that moment.
		try:
			self.inputs_queue.put_nowait(frame)
			return True
		except queue.Full:
			return False

	def pull_result(self) -> Tuple[bool, Tuple[np.ndarray, np.ndarray]]:
		# Non blocking results pull method. Returns tuple (status, (frame, result))
		# status == True informs that pull was successful
		# status == False informs that there are no results to be pulled
		# (frame, result) tuple is the actual element pulled from the results queue
		# For status == False, it's set to (None, None) as it's N/A 
		try:
			return True, self.results_queue.get_nowait()
		except queue.Empty:
			return False, (None, None)

	def _inference_pipeline_thread(self):	
		num_inference_executors = len(self.inference_executors)

		# In the first iteration only fill executors inputs
		i = 0
		while i < num_inference_executors:
			try:
				input = self.inputs_queue.get(timeout=1)
				self.inference_executors[i].input_queue.put(input)
			except queue.Empty:
				if self.exit_event.is_set():
					return
				continue
			i += 1

		inference_pipeline_handler = InferencePipelineHandler(self.inputs_queue, self.results_queue)
		initial_pipeline_step = InferencePipelineHandler.PipelineStep.PULL_RESULT
		execution_result = InferencePipelineHandler.ExecutionResult(initial_pipeline_step, None)
		i = 0
		while not self.exit_event.is_set():
			inference_executor = self.inference_executors[i]
			previous_execution_result = execution_result
			execution_result = inference_pipeline_handler.run_inference_pipeline(inference_executor, previous_execution_result)
			if execution_result.pipeline_step == InferencePipelineHandler.PipelineStep.FINISHED:
				i = (i + 1) % num_inference_executors
				execution_result.pipeline_step = initial_pipeline_step

	def _inference_executors_monitoring_thread(self):
		while not self.exit_event.is_set():
			time.sleep(1)
			if not self.abort_event.is_set():
				for inference_executor in self.inference_executors:
					if inference_executor.abort_event.is_set():
						self.logger.info(f"Received abort signal from Inference-Executor-{inference_executor.id}. Notifying Stream Analyzer...")
						self.abort_event.set()
