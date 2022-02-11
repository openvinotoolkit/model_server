### LICENSE HEADER ###

import argparse
from stream_reader import StreamReader
from inference_executor import InferenceExecutor
from io_processor import IOProcessor
from http_visualizer import HttpVisualizer
from stream_analyzer import StreamAnalyzer
from logger import LoggerConfig
from use_cases.person_vehicle_bike_detection import PersonVehicleBikeDetection


def get_config():
	parser = argparse.ArgumentParser(description="Client for real time video stream analysis")
	parser.add_argument("--stream_url", required=True,  help="URL to the video stream. Example: rtsp://localhost:8888/stream")
	parser.add_argument("--ovms_url", required=True, help="Address of OVMS gRPC endpoint. Example: localhost:9000")
	parser.add_argument("--model_name", required=True, help="Name of the target model.")
	parser.add_argument("--model_version", required=False, default=0, help="Version of the taget model. Default: latest available")
	parser.add_argument("--visualizer_port", required=False, help="Port of the inferece results visualizer webservice. "
																  "If not specified, visualizer will not be launched")
	parser.add_argument("--log_level", required=False, default="INFO", help="Logging level - available values: INFO, DEBUG, ERROR")

	args = vars(parser.parse_args())
	return args

def main():
	config = get_config()
	stream_url = config["stream_url"]
	visualizer_port = config["visualizer_port"]
	ovms_url = config["ovms_url"]
	model_name = config["model_name"]
	model_version = config["model_version"]
	log_level = config["log_level"]

	LoggerConfig.set_log_level(log_level)

	launch_http_visualizer = False
	if visualizer_port:
		launch_http_visualizer = True

	stream_reader = StreamReader(stream_url)
	inference_executor = InferenceExecutor(ovms_url, model_name, model_version)
	io_processor = IOProcessor(PersonVehicleBikeDetection, launch_http_visualizer)
	if launch_http_visualizer:
		http_visualizer = HttpVisualizer(visualizer_port, io_processor)
	else:
		http_visualizer = None

	stream_analyzer = StreamAnalyzer(stream_reader, inference_executor, io_processor, http_visualizer)
	stream_analyzer.run()

if __name__ == "__main__":
	main()