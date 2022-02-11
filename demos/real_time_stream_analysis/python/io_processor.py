import cv2
from queue import Queue
from use_cases.use_case import UseCase
from logger import get_logger

class IOProcessor:

    def __init__(self, use_case: UseCase, launch_http_visualizer: bool):
        self.logger = get_logger(__name__)
        self.use_case = use_case
        self.enable_visualization = launch_http_visualizer
        self.visualizer_frames_queue = Queue()
        if self.enable_visualization and not self.use_case.supports_visualization():
            raise ValueError("Cannot enable visualization for that use case")

    def preprocess(self, data):
        return self.use_case.preprocess(data)

    def postprocess(self, frame, inference_result):
        if self.enable_visualization:
            frame = self.use_case.visualize(frame, inference_result)
            self.visualizer_frames_queue.put(frame)
        self.use_case.postprocess(inference_result)

    def get_visualizer_frame(self):
        if self.enable_visualization:
            frame = self.visualizer_frames_queue.get()
            return True, frame
        else:
            raise RuntimeError("Visualization has not been enabled for that use case")
