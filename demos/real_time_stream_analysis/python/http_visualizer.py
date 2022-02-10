from flask import Flask, render_template, Response
from logger import get_logger
import cv2
import time

logger = get_logger(__name__)

class HttpVisualizer:
	def __init__(self, visualizer_port, postprocessor):
		self.port = visualizer_port
		self.flask_server = Flask(__name__)
		self.postprocessor = postprocessor


	def read_postprocessed_frame(self):
		while True:
			time.sleep(0.04)
			success, frame = self.postprocessor.get_postprocessed_frame()
			if not success:
				logger.info("HTTP Visualizer broke")
				break
			logger.debug("Visualizing frame")
			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()
			yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


	def run(self):
		logger.info("Starting HTTP Visualizer")
		#flask_server = Flask(__name__)

		@self.flask_server.route('/video_feed')
		def video_feed():
			return Response(self.read_postprocessed_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


		@self.flask_server.route('/')
		def index():
			return render_template('index.html')

		self.flask_server.run(host="0.0.0.0", port=self.port, debug=False)
