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

from abc import ABC, abstractmethod
import datetime
import json

import falcon
import numpy as np

from src.logger import get_logger
from src.preprocessing.preprocess_image import preprocess_binary_image as default_preprocessing
from src.api.ovms_connector import OvmsUnavailableError, ModelNotFoundError

logger = get_logger(__name__)


class Model(ABC):
    def __init__(self, endpoint, ovms_connector, input_configs, output_configs):
        self.endpoint = endpoint
        self.ovms_connector = ovms_connector
        self.input_configs = input_configs
        self.output_configs = output_configs
        self.labels = {output_name:
                       {index: label for label,
                           index in self.output_configs[output_name].classes.items()}
                       for output_name in self.output_configs.keys()}

    def preprocess_binary_image(self, binary_image: bytes) -> np.ndarray:
        try:
            # Assuming single input for now
            preprocessing_config = next(iter(self.input_configs.values()))
            preprocessing_config = preprocessing_config.as_preprocessing_options()
            preprocessed_image = default_preprocessing(
                binary_image, **preprocessing_config)
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        except Exception as e:
            logger.exception('Failed to preprocess binary image')
            raise
        return preprocessed_image

    @abstractmethod
    def postprocess_inference_output(self, inference_output: dict) -> str:
        # Model specific code
        return

    def on_post(self, req, resp):
        # Main flow for the inference request

        # Retrieve request headers as python dictionary
        request_headers = req.headers
        logger.debug(f"Received request with headers: {request_headers}")
        # Retrieve raw bytes from the request
        request_body = req.bounded_stream.read()

        # Preprocess request body
        try:
            preprocessing_start_time = datetime.datetime.now()
            input_image = self.preprocess_binary_image(request_body)
            duration = (datetime.datetime.now() -
                        preprocessing_start_time).total_seconds() * 1000
            logger.debug(f"Input preprocessing time: {duration} ms")
        except Exception as ex:
            logger.exception("Failed request preprocessing")
            body = {"message": str(ex)}
            resp.status = falcon.HTTP_400
            resp.body = json.dumps(body)
            return

        # Send inference request to corresponding model in OVMS
        try:
            connection_start_time = datetime.datetime.now()
            # Assuming models with just one input for now
            inference_input = {next(iter(self.input_configs)): input_image}
            inference_output = self.ovms_connector.send(inference_input)
            duration = (datetime.datetime.now() -
                        connection_start_time).total_seconds() * 1000
            logger.debug(f"OVMS request handling time: {duration} ms")
        except (ValueError, TypeError) as ex:
            logger.exception("Invalid request data")
            body = {"message": str(ex)}
            resp.status = falcon.HTTP_400
            resp.body = json.dumps(body)
            return
        except ModelNotFoundError as ex:
            logger.exception("Model not found")
            body = {"message": str(ex)}
            resp.status = falcon.HTTP_500
            resp.body = json.dumps(body)
            return
        except OvmsUnavailableError as ex:
            logger.exception("OVMS unavailable")
            body = {"message": str(ex)}
            resp.status = falcon.HTTP_503
            resp.body = json.dumps(body)
            return
        except Exception as ex:
            logger.exception("Internal OVMS error")
            body = {"message": str(ex)}
            resp.status = falcon.HTTP_500
            resp.body = json.dumps(body)
            return

        # Postprocess
        try:
            postprocessing_start_time = datetime.datetime.now()
            results = self.postprocess_inference_output(inference_output)
            duration = (datetime.datetime.now() -
                        postprocessing_start_time).total_seconds() * 1000
            logger.debug(f"Output postprocessing time: {duration} ms")
        except Exception as ex:
            logger.exception("Error during request postprocessing")
            body = {"message": str(ex)}
            resp.status = falcon.HTTP_500
            resp.body = json.dumps(body)
            return

        # If model did not found results
        if results is None:
            resp.status = falcon.HTTP_204
            return

        # Send response back
        resp.status = falcon.HTTP_200
        resp.body = results
        return
