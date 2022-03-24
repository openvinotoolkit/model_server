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

import time
from logger import get_logger
from utils import signal_capture
from processing_supervisor import ProcessingSupervisor

class StreamAnalyzer:	

	def __init__(self, stream_reader, inference_manager, io_processor, http_visualizer):
		self.logger = get_logger(__name__)
		self.stream_reader = stream_reader
		self.inference_manager = inference_manager
		self.io_processor = io_processor
		self.http_visualizer = http_visualizer

		self.supervisor = ProcessingSupervisor(self.logger)

	def cleanup(self):
		self.logger.info("Shutting down Stream Analyzer...")
		self.inference_manager.shutdown()
		if self.http_visualizer:
			self.http_visualizer.shutdown()
		self.logger.info("Stream Analyzer shut down successfully")

	def run(self):
		self.logger.info("Starting Stream Analyzer")
		self.inference_manager.initialize()
		if self.http_visualizer:
			self.http_visualizer.initialize()

		self.supervisor.set_check_start_time()

		while not signal_capture.kill:

			if self.inference_manager.abort_event.is_set():
				self.logger.info("Received abort signal from Inference Manager. Shutting down...")
				break

			if self.supervisor.is_check_possible():
				self.supervisor.set_check_end_time()
				stream_reader_status = self.supervisor.check_stream_reader_health()
				if stream_reader_status == ProcessingSupervisor.StreamReaderStatus.RETRY:
					self.logger.info(f"[Retry {self.supervisor.retries_counter}/{ProcessingSupervisor.Config._RETRIES_LIMIT}] " 
									 f"Waiting {ProcessingSupervisor.Config._STREAM_RECOVERY_WAIT_SECONDS}s before reinitializing the stream...")
					time.sleep(ProcessingSupervisor.Config._STREAM_RECOVERY_WAIT_SECONDS)
					if self.stream_reader.initialize():
						self.supervisor.reset_retries_counter()
					self.supervisor.reset_perf_counters()
					self.supervisor.set_check_start_time()
					continue
				elif stream_reader_status == ProcessingSupervisor.StreamReaderStatus.CLOSE:
					self.logger.error(f"Max retries limit has been reached. Please check stream health. Application will shutdown now.")
					break

				ok = self.supervisor.check_processing_pipeline_health()
				# TO DO: Decide if any action is needed in case of permanent input buffer overflow

				self.supervisor.report_performance()
				self.supervisor.reset_perf_counters()
				self.supervisor.set_check_start_time()

			self.supervisor.increase_iterator()
			
			success, frame = self.stream_reader.get_frame()
			if not success:
				self.supervisor.increase_failed_read_frames_counter()
				continue
			self.supervisor.increase_frames_counter()

			success, input = self.io_processor.preprocess(frame)
			if not success:
				self.logger.info("Preprocessing failed. Stream Analyzer is not functional and can't recover. Shutting down...")
				break

			success = self.inference_manager.schedule_inference(input)
			if not success:
				self.supervisor.increase_dropped_frames_counter()
				continue

			success, (frame, result) = self.inference_manager.pull_result()
			if not success:
				continue

			success = self.io_processor.postprocess(frame, result)
			if not success:
				self.logger.info("Postprocessing failed. Stream Analyzer is not functional and can't recover. Shutting down...")
				break

			self.supervisor.increase_processed_frames_counter()
		self.cleanup()
