import cv2
import threading
import time
from logger import get_logger

logger = get_logger(__name__)

class StreamAnalyzer:

	def __init__(self, stream_reader, inference_executor, postprocessor, http_visualizer):
		self.stream_reader = stream_reader
		self.inference_executor = inference_executor
		self.postprocessor = postprocessor
		self.http_visualizer = http_visualizer

	def run(self):
		logger.info("Starting Stream Analyzer")
		if self.http_visualizer:
			threading.Thread(target=self.http_visualizer.run).start()

		frame_counter = 0
		start_time = time.time()
		while True:
			success, frame = self.stream_reader.get_frame()
			if not success:
				break
			result = self.inference_executor.predict(frame)
			self.postprocessor.postprocess(frame, result)
			if frame_counter % 100 == 0:
				end_time = time.time()
				duration = end_time - start_time
				fps = int(100 / duration)
				logger.info("Frame: {}; FPS: {}".format(frame_counter, fps))
				start_time = end_time
			frame_counter += 1

