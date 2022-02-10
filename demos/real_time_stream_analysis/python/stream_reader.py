import cv2
from logger import get_logger

logger = get_logger(__name__)

class StreamReader:

	def __init__(self, stream_url):
		self.stream = cv2.VideoCapture(stream_url)  
		logger.info("Stream Reader initialized successfully")

	def get_frame(self):
		success, frame = self.stream.read()
		if not success:
			raise ValueError
		logger.debug("Captured stream frame")
		return success, frame
			