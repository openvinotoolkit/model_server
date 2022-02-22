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

import numpy as np
import queue
from typing import Tuple

class InferencePipelineHandler:

	class PipelineStep:
			PULL_RESULT = 1
			PUSH_RESULT = 2
			PULL_INPUT = 3
			PUSH_INPUT = 4
			FINISHED = 5

	class ExecutionResult:
		def __init__(self, pipeline_step, pipeline_data = None) -> None:
			self.pipeline_step = pipeline_step
			self.pipeline_data = pipeline_data

	def __init__(self, inputs_queue, results_queue) -> None:
		self.inputs_queue = inputs_queue
		self.results_queue = results_queue

	def _get_result_from_executor(self, inference_executor) -> Tuple[bool, np.ndarray]:
		try:
			latest_result = inference_executor.result_queue.get(timeout=1)
			return True, latest_result
		except queue.Empty:
			return False, None

	def _push_result_to_main_results_queue(self, result) -> bool:
		try:
			self.results_queue.put(result, timeout=1)
			return True
		except queue.Full:
			return False

	def _get_input_from_main_inputs_queue(self) -> Tuple[bool, np.ndarray]:
		try:
			input = self.inputs_queue.get(timeout=1)
			return True, input
		except queue.Empty:
			return False, None

	def _push_input_to_executor(self, inference_executor, input) -> bool:
		try:
			inference_executor.input_queue.put(input, timeout=1)
			return True
		except queue.Full:
			return False		

	def run_inference_pipeline(self, inference_executor, previous_execution_result) -> ExecutionResult:

		current_step = previous_execution_result.pipeline_step
		pipeline_data = previous_execution_result.pipeline_data

		result = None
		input = None

		if current_step == InferencePipelineHandler.PipelineStep.PULL_RESULT:
			success, result = self._get_result_from_executor(inference_executor)
			if not success:
				return InferencePipelineHandler.ExecutionResult(current_step, None)
			else:
				current_step += 1

		if current_step == InferencePipelineHandler.PipelineStep.PUSH_RESULT:
			if result is None:
				result = pipeline_data
			success = self._push_result_to_main_results_queue(result)
			if not success:
				return InferencePipelineHandler.ExecutionResult(current_step, result)
			else:
				current_step += 1
		
		if current_step == InferencePipelineHandler.PipelineStep.PULL_INPUT:
			success, input = self._get_input_from_main_inputs_queue()
			if not success:
				return InferencePipelineHandler.ExecutionResult(current_step, None)
			else:
				current_step += 1

		if current_step == InferencePipelineHandler.PipelineStep.PUSH_INPUT:
			if input is None:
				input = pipeline_data
			success = self._push_input_to_executor(inference_executor, input)
			if not success:
				return InferencePipelineHandler.ExecutionResult(current_step, input)
			else:
				current_step += 1
			
		return InferencePipelineHandler.ExecutionResult(current_step, None)
