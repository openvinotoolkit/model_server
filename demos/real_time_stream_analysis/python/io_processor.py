from typing import Tuple
import numpy as np
import queue
from use_cases.use_case import UseCase
from logger import get_logger
from processing_supervisor import ProcessingSupervisor
import multiprocessing

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

    def preprocess(self, data):
        return self.use_case.preprocess(data)

    def postprocess(self, frame, inference_result):
        if self.enable_visualization:
            frame = self.use_case.visualize(frame, inference_result)
            try:
                self.visualizer_frames_queue.put_nowait(frame)
            except queue.Full:
                #self.logger.debug("Visualizer buffer full. Dropping frame...")
                self.dropped_visualizer_frames_counter += 1
        self.use_case.postprocess(inference_result)

        self.postprocessed_frames_counter += 1
        if self.postprocessed_frames_counter % self.LOGGING_INTERVAL == 0:
            if self.dropped_visualizer_frames_counter > self.LOGGING_INTERVAL/2:
                self.logger.warning(f"Out of {self.LOGGING_INTERVAL} last inference results, "
                                     "more than half have not been visualized due to full visualizer buffer")
            self.postprocessed_frames_counter = 0
            self.dropped_visualizer_frames_counter = 0

