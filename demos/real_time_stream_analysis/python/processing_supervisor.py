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

import enum
import time

class ProcessingSupervisor:
	
	class Config:
		_CHECK_INTERVAL = 100
		_RETRIES_LIMIT = 6
		_STREAM_RECOVERY_WAIT_SECONDS = 10

	class StreamReaderStatus(enum.Enum):
		OK = 1
		RETRY = 2
		CLOSE = 3

	retries_counter = 0
	iterator = 0
	frames_counter = 0
	failed_read_frames_counter = 0
	dropped_frames_counter = 0
	processed_frames_counter = 0

	check_start_time = None
	check_end_time = None

	def __init__(self, logger) -> None:
		self.logger = logger

	def increase_retries_counter(self) -> None:
		self.retries_counter += 1

	def increase_iterator(self) -> None:
		self.iterator += 1

	def increase_frames_counter(self) -> None:
		self.frames_counter += 1

	def increase_failed_read_frames_counter(self) -> None:
		self.failed_read_frames_counter += 1

	def increase_dropped_frames_counter(self) -> None:
		self.dropped_frames_counter += 1

	def increase_processed_frames_counter(self) -> None:
		self.processed_frames_counter += 1

	def set_check_start_time(self) -> None:
		self.check_start_time = time.time()

	def set_check_end_time(self) -> None:
		self.check_end_time = time.time()

	def is_check_possible(self) -> bool:
		if self.iterator % self.Config._CHECK_INTERVAL == 0 and self.iterator != 0:
			return True
		return False

	def check_stream_reader_health(self) -> StreamReaderStatus:
		if self.failed_read_frames_counter > self.iterator/2:
			self.logger.warning(f"Out of {self.iterator} attempts to read frame from the stream, more than half have been unsuccessful")
			if self.retries_counter < self.Config._RETRIES_LIMIT:
				self.retries_counter += 1
				return ProcessingSupervisor.StreamReaderStatus.RETRY
			else:
				self.retries_counter = 0
				return ProcessingSupervisor.StreamReaderStatus.CLOSE
		return ProcessingSupervisor.StreamReaderStatus.OK
	
	def check_processing_pipeline_health(self) -> bool:
		if self.dropped_frames_counter > self.frames_counter/2:
			self.logger.warning(f"Out of {self.frames_counter} received frames, more than half have been dropped due to full input buffer. "
								"This indicates that stream speed exceeds inference execution capability")
			return False
		return True
					
	def report_performance(self) -> None:
		duration = self.check_end_time - self.check_start_time
		fps = int(self.processed_frames_counter / duration)
		self.logger.info(f"For {self.frames_counter} received frames - FPS: {fps}; Frames dropped: {self.dropped_frames_counter};")

	def reset_perf_counters(self) -> None:
		self.iterator = 0
		self.frames_counter = 0
		self.failed_read_frames_counter = 0
		self.dropped_frames_counter = 0
		self.processed_frames_counter = 0
	
	def reset_retries_counter(self) -> None:
		self.retries_counter = 0
