import falcon
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

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
        #TODO: decoding logic
        return image

    @abstractmethod
    def postprocess_inference_output(self, inference_output: dict):
        # Model specific code
        return

    def on_post(self, req, resp):
        # Main flow for the inference request

        # Retrieve request headers as python dictionary 
        request_headers = req.headers
        # Retrieve raw bytes from the request
        request_body = req.bounded_stream.read()

        print(f"Request headers: \n{request_headers}")

        # Preprocess request body
        input_image = self.preprocess_binary_image(request_body)

        # Send inference request to corresponding model in OVMS
        inference_ouput = self.ovms_connector.send(input_image)

        # Postprocess
        results = self.postprocess_inference_output(inference_ouput)

        # Send response back
        resp.status = falcon.HTTP_200
        resp.body = results
        return


