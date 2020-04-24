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

    def preprocess_binary_image(self, binary_image: bytes, channels: int = None,
                                dtype=tf.dtypes.uint8, scale: float = None,
                                standardization=False,
                                reverse_input_channels=False) -> np.ndarray:
        try: 
            # Validate image size, if needed, resize to dimensions required by target model
            binary_image = self.resize_image(binary_image)
            # Decode image to get it as numpy array
            decoded_image = self.decode_image(binary_image, channels, dtype, scale, 
                                                standardization, reverse_input_channels)
        except Exception as e:
            # TODO: Error handling
            return
        return decoded_image

    def resize_image(binary_image: bytes) -> bytes:
        input_shape = self.ovms_connector.input_shape
        # TODO: resizing logic
        return binary_image

    def decode_image(self, binary_image: bytes, channels: int = None,
                    dtype=tf.dtypes.uint8, scale: float = None,
                    standardization=False,
                    reverse_input_channels=False) -> np.ndarray:
        # TODO: decoding logic
        return image

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


