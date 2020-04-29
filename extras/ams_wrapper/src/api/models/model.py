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

import datetime
import json
import falcon
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from logger import get_logger

logger = get_logger(__name__)

class Model(ABC):

    def __init__(self, ovms_connector):
        self.ovms_connector = ovms_connector
        self.model_name = None # subtype / model name in AMS
        self.result_type = None # type class
        self.labels = None # load_labels

    def load_labels(labels_path: str) -> str:
        try:
            with open(labels_path, 'r') as labels_file:
                data = json.load(labels_file)
        except Exception as e:
            logger.exception("Error occurred while opening labels file: {}".format(e))
            sys.exit(1)
        return data

    def preprocess_binary_image(self, binary_image: bytes) -> np.ndarray:
        # By default the only performed preprocessing is converting image 
        # from binary format to numpy ndarray. If a model requires more specific 
        # preprocessing this method should be implemented in its class.
        try: 
            """
            # Perform default preprocessing
            preprocessed_image = default_preprocessing(binary_image)
            """
            preprocessed_image = None
        except Exception as e:
            # TODO: Error handling
            return
        return preprocessed_image

    @abstractmethod
    def postprocess_inference_output(self, inference_output: dict) -> str:
        # Model specific code
        return

    def on_post(self, req, resp):
        # Main flow for the inference request

        # TODO: Handle errors

        # Retrieve request headers as python dictionary 
        request_headers = req.headers
        logger.debug(f"Received request with headers: {request_headers}")
        # Retrieve raw bytes from the request
        request_body = req.bounded_stream.read()

        # Preprocess request body
        preprocessing_start_time = datetime.datetime.now()
        input_image = self.preprocess_binary_image(request_body)
        duration = (datetime.datetime.now() -
                    preprocessing_start_time).total_seconds() * 1000
        logger.debug(f"Input preprocessing time: {duration} ms")

        # Send inference request to corresponding model in OVMS
        connection_start_time = datetime.datetime.now()
        inference_ouput = self.ovms_connector.send(input_image)
        duration = (datetime.datetime.now() -
                    connection_start_time).total_seconds() * 1000
        logger.debug(f"OVMS request handling time: {duration} ms")

        # Postprocess
        postprocessing_start_time = datetime.datetime.now()
        results = self.postprocess_inference_output(inference_ouput)
        duration = (datetime.datetime.now() -
                    postprocessing_start_time).total_seconds() * 1000
        logger.debug(f"Output postprocessing time: {duration} ms")

        # Send response back
        resp.status = falcon.HTTP_200
        resp.body = results
        return


