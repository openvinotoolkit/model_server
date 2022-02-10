### LICENSE HEADER ###

import argparse
from stream_reader import StreamReader
from inference_executor import InferenceExecutor
from postprocessor import Postprocessor
from http_visualizer import HttpVisualizer
from stream_analyzer import StreamAnalyzer
from logger import get_logger

logger = get_logger(__name__)

def my_custom_postprocessing(result):
	CLASSES = ["None", "Pedestrian", "Vehicle", "Bike", "Other"]
	CONFIDENCE_THRESHOLD = 0.75
	for batch, data in enumerate(result):
		pred = data[0]
		for values in enumerate(pred):
				index = values[0]
				l_pred = values[1]

				img_id = l_pred[0]
				label = l_pred[1]
				conf = l_pred[2]

				if label > 4:
					label = 4

				if img_id != -1 and conf >= CONFIDENCE_THRESHOLD:
					if CLASSES[int(label)] == "Pedestrian":
						logger.warning("Detected pedestrian on the road") 

def get_config():
	parser = argparse.ArgumentParser(description="Client for real time video stream analysis")
	parser.add_argument("--stream_url", required=True,  help="URL to the video stream. Example: rtsp://localhost:8888/stream")
	parser.add_argument("--ovms_url", required=True, help="Address of OVMS gRPC endpoint. Example: localhost:9000")
	parser.add_argument("--model_name", required=True, help="Name of the target model.")
	parser.add_argument("--model_version", required=False, default=0, help="Version of the taget model. Default: latest available")
	parser.add_argument("--visualizer_port", required=False, help="Port of the inferece results visualizer webservice. "
																  "If not specified, visualizer will not be launched")

	args = vars(parser.parse_args())
	return args

def main():
	config = get_config()
	stream_url = config["stream_url"]
	visualizer_port = config["visualizer_port"]
	ovms_url = config["ovms_url"]
	model_name = config["model_name"]
	model_version = config["model_version"]

	stream_reader = StreamReader(stream_url)
	inference_executor = InferenceExecutor(ovms_url, model_name, model_version)
	postprocessor = Postprocessor(enable_visualization=False)
	http_visualizer = None
	if visualizer_port:
		postprocessor = Postprocessor(enable_visualization=True)
		http_visualizer = HttpVisualizer(visualizer_port, postprocessor)

	postprocessor.add_postprocessing_routine("Pedestrian alert", my_custom_postprocessing)
	stream_analyzer = StreamAnalyzer(stream_reader, inference_executor, postprocessor, http_visualizer)
	stream_analyzer.run()

if __name__ == "__main__":
	main()