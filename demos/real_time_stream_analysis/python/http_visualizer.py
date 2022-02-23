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

import multiprocessing
import os
import queue
from flask import Flask, render_template, Response
from logger import get_logger
import cv2
import signal

class HttpVisualizer():

	BIND_ADDRESS="localhost" # nosec

	def __init__(self, visualizer_port, buffer_size):
		self.logger = get_logger(__name__)
		self.port = visualizer_port
		flask_server = self._make_flask_server()
		self.flask_process = multiprocessing.Process(target=flask_server.run, kwargs={"host": self.BIND_ADDRESS,"port": self.port, "debug":False})
		self.frames_queue = multiprocessing.Queue(maxsize=buffer_size)
		self.logger.info(f"Visualizer frames buffer capacity set to {buffer_size} frames")

	def initialize(self):
		self.logger.info("Starting Flask webserver...")
		self.flask_process.start()

	def shutdown(self):
		self.logger.info("Shutting down HTTP Visualizer...")
		self.logger.info("Sending termination singal to Flask web server...")
		os.kill(self.flask_process.pid, signal.SIGKILL)
		self.flask_process.join()
		self.logger.info("Flask web server terminated successfully")
		self.logger.info("Flushing visualizer input buffer")
		self._flush_queue()
		self.logger.info("HTTP Visualizer shut down successfully")

	def _flush_queue(self):
		while not self.frames_queue.empty():
			self.frames_queue.get()

	def get_frames_queue(self) -> multiprocessing.Queue:
		return self.frames_queue

	def read_frame(self):
		while True:
			try:
				frame = self.frames_queue.get_nowait()
			except queue.Empty:
				continue
			success, buffer = cv2.imencode('.jpg', frame)
			if not success:
				continue
			frame = buffer.tobytes()
			yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n") 


	def _make_flask_server(self):
		flask_server = Flask(__name__)

		@flask_server.route("/video_feed")
		def video_feed():
			return Response(self.read_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")


		@flask_server.route("/")
		def index():
			return render_template("index.html")
		
		return flask_server
