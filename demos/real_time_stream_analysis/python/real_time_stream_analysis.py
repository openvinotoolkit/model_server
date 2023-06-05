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

import argparse
from stream_reader import StreamReader
from inference_manager import InferenceManager
from io_processor import IOProcessor
from http_visualizer import HttpVisualizer
from stream_analyzer import StreamAnalyzer
from logger import LoggerConfig, get_logger
from use_cases import PersonVehicleBikeDetection

logger = get_logger(__name__)

def get_config():
	parser = argparse.ArgumentParser(description="Client for real time video stream analysis")
	parser.add_argument("--stream_url", required=True, type=str, help="URL to the video stream. Example: rtsp://localhost:8888/stream")
	parser.add_argument("--ovms_url", required=True, type=str, help="Address of OVMS gRPC endpoint. Example: localhost:9000")
	parser.add_argument("--model_name", required=True, type=str, help="Name of the target model")
	parser.add_argument("--model_version", required=False, type=int, default=0, help="Version of the target model. Default: latest available")
	parser.add_argument("--visualizer_port", required=False, type=int, help="Port of the inference results visualizer web service. "
																  			"If not specified, visualizer will not be launched")
	parser.add_argument("--binary_input", required=False, action="store_true", help="Convert frames to binary format before sending them to OVMS. Default: False")
	parser.add_argument("--inference_executors", required=False, type=int, default=4, help="Number of inference executor threads. Default: 4")
	parser.add_argument("--buffer_size", required=False, type=int, default=100, help="Number of frames the have been received from the stream and are awaiting inference or visualization. "
																					  "When buffer is full, frames are dropped. Default: 100")
	parser.add_argument("--log_level", required=False, type=str, default="INFO", help="Logging level - available values: INFO, DEBUG, ERROR. Default: INFO")

	args = vars(parser.parse_args())
	return args

def main():

	config = get_config()
	stream_url = config["stream_url"]
	visualizer_port = config["visualizer_port"]
	ovms_url = config["ovms_url"]
	model_name = config["model_name"]
	model_version = config["model_version"]
	binary_input = config["binary_input"]
	inference_executors = config["inference_executors"]
	buffer_size = config["buffer_size"]
	log_level = config["log_level"]

	LoggerConfig.set_log_level(log_level)

	launch_http_visualizer = False
	if visualizer_port:
		launch_http_visualizer = True


	stream_reader = StreamReader(stream_url)
	success = stream_reader.initialize()
	if not success:
		return
	
	inference_manager = InferenceManager(ovms_url, model_name, model_version, inference_executors, binary_input, buffer_size)
	if launch_http_visualizer:
		http_visualizer = HttpVisualizer(visualizer_port, buffer_size)
	else:
		http_visualizer = None

	visualizer_frames_queue = http_visualizer.get_frames_queue() if http_visualizer is not None else None
	try:
		io_processor = IOProcessor(PersonVehicleBikeDetection, visualizer_frames_queue)
	except Exception as error:
		logger.error(f"Error occurred during creating IOProcessor: {str(error)}")
		logger.info(f"Stream Analyzer cannot be created. Shutting down...")
		return
	stream_analyzer = StreamAnalyzer(stream_reader, inference_manager, io_processor, http_visualizer)
	stream_analyzer.run()

if __name__ == "__main__":
	main()
