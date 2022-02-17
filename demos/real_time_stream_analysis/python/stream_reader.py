from typing import Tuple
import cv2
import numpy as np
from logger import get_logger


class StreamReader:

	def __init__(self, stream_url):
		self.logger = get_logger(__name__)
		self.stream_url =stream_url

	def get_frame(self) -> Tuple[bool, np.ndarray]:
		try:
			return self.stream.read()
		except:
			raise RuntimeError("Stream object broken or not initialized")

	def initialize(self) -> bool:
		self.logger.info(f"Initializing Stream Reader for stream: {self.stream_url} ...")
		self.stream = cv2.VideoCapture(self.stream_url)
		if self.stream.isOpened():
			self.logger.info("Stream Reader initialized successfully")
			return True
		else:
			self.logger.error("Failed to initialize Stream Reader")
			return False
