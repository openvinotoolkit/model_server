import threading
import time
from logger import get_logger

class StreamAnalyzer:

	def __init__(self, stream_reader, inference_executor, io_processor, http_visualizer):
		self.logger = get_logger(__name__)
		self.stream_reader = stream_reader
		self.inference_executor = inference_executor
		self.io_processor = io_processor
		self.http_visualizer = http_visualizer

	def run(self):
		self.logger.info("Starting Stream Analyzer")
		if self.http_visualizer:
			threading.Thread(target=self.http_visualizer.run).start()

		frames_counter = 0
		dropped_frames_counter = 0
		processed_frames_counter = 0
		start_time = time.time()
		while True:

			if frames_counter % 100 == 0 and frames_counter != 0:
				end_time = time.time()
				duration = end_time - start_time
				fps = int(processed_frames_counter / duration)
				self.logger.info("For 100 received frames - FPS: {}; Frames dropped: {};".format(fps, dropped_frames_counter))
				start_time = end_time
				frames_counter = 0
				processed_frames_counter = 0
				dropped_frames_counter = 0

			frames_counter += 1
			
			success, frame = self.stream_reader.get_frame()
			if not success:
				break

			input = self.io_processor.preprocess(frame)
			success = self.inference_executor.schedule_inference(input)
			if not success:
				self.logger.debug("Input buffer is full. Dropping frame...")
				dropped_frames_counter += 1
				continue

			frame, result = self.inference_executor.pull_result()
			if result is None:
				continue

			self.io_processor.postprocess(frame, result)
			processed_frames_counter += 1

