import cv2
from logger import get_logger


class StreamReader:

	def __init__(self, stream_url):
		self.logger = get_logger(__name__)
		self.stream = cv2.VideoCapture(stream_url)  
		self.logger.info("Stream Reader initialized successfully")

	def get_frame(self):
		success, frame = self.stream.read()
		if not success:
			raise ValueError
		return success, frame
			