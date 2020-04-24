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
from cheroot.wsgi import Server as WSGIServer, PathInfoDispatcher
from api.dispatcher import create_dispatcher
from config import AVAILABLE_MODELS
from logger import get_logger

logger = get_logger(__name__)

def start_rest_service(port, num_threads):
    dispatcher = PathInfoDispatcher({'/': create_dispatcher(AVAILABLE_MODELS)})
    server = WSGIServer(('0.0.0.0', port), dispatcher,
                        numthreads=num_threads)
    logger.info(f"Server will start listetning on port {port}")
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, help='Listening port',
                          required=False, default=5000)
    parser.add_argument('--workers', type=int, help='Number of service workers',
                          required=False, default=1)
    args = parser.parse_args()
    start_rest_service(args.port, args.workers)
main()
