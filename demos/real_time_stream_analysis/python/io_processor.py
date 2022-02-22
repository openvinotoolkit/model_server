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
from use_cases.use_case import UseCase
from logger import get_logger
from processing_supervisor import ProcessingSupervisor
import multiprocessing
import numpy as np

class IOProcessor:

    LOGGING_INTERVAL = ProcessingSupervisor.Config._CHECK_INTERVAL
    postprocessed_frames_counter = 0
    dropped_visualizer_frames_counter = 0

    def __init__(self, use_case: UseCase, visualizer_frames_queue: multiprocessing.Queue):
        self.logger = get_logger(__name__)
        self.logger.info("Initializing IO Processor")
        self.use_case = use_case
        self.visualizer_frames_queue = visualizer_frames_queue
        self.enable_visualization = False if visualizer_frames_queue is None else True
        if self.enable_visualization and not self.use_case.supports_visualization():
            raise ValueError("Cannot enable visualization for that use case")

        self.logger.info("IO Processor initialized successfully")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        return self.use_case.preprocess(frame)

    def postprocess(self, frame: np.ndarray, inference_result: np.ndarray):
        if self.enable_visualization:
            frame = self.use_case.visualize(frame, inference_result)
            try:
                self.visualizer_frames_queue.put_nowait(frame)
            except queue.Full:
                self.dropped_visualizer_frames_counter += 1
        self.use_case.postprocess(inference_result)

        self.postprocessed_frames_counter += 1
        if self.postprocessed_frames_counter % self.LOGGING_INTERVAL == 0:
            if self.dropped_visualizer_frames_counter > self.LOGGING_INTERVAL/2:
                self.logger.warning(f"Out of {self.LOGGING_INTERVAL} last inference results, "
                                     "more than half have not been visualized due to full visualizer buffer")
            self.postprocessed_frames_counter = 0
            self.dropped_visualizer_frames_counter = 0
