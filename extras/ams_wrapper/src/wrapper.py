#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import glob

from cheroot.wsgi import Server as WSGIServer, PathInfoDispatcher
from src.api.dispatcher import create_dispatcher
from src.logger import get_logger

logger = get_logger(__name__)

AMS_MODELS_PATH = "/opt/ams_models"

def get_available_models(models_path=AMS_MODELS_PATH):
    pattern = "{}/*_model.json".format(AMS_MODELS_PATH)
    available_models = glob.glob(pattern)
    if not available_models:
        raise FileNotFoundError("No AMS model configuration has been found in: {}".format(AMS_MODELS_PATH))
    return available_models

def start_rest_service(port, num_threads, ovms_port):
    available_models = get_available_models()
    dispatcher = PathInfoDispatcher({'/': create_dispatcher(available_models, ovms_port)})
    server = WSGIServer(('0.0.0.0', port), dispatcher,
                        numthreads=num_threads)
    logger.info(f"AMS service will start listening on port {port}")
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, help='AMS service listening port',
                          required=False, default=5000)
    parser.add_argument('--workers', type=int, help='Number of service workers',
                          required=False, default=1)
    parser.add_argument('--ovms_port', type=int, help='OpenVINO Model Server port',
                          required=False, default=9000)
    args = parser.parse_args()
    start_rest_service(args.port, args.workers, args.ovms_port)

main()
