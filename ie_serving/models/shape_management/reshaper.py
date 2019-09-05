#
# Copyright (c) 2019 Intel Corporation
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
import json

import falcon
import datetime

from ie_serving.logger import get_logger
from ie_serving.models.shape_management.utils import ShapeMode
from ie_serving.server.constants import GRPC, REST
from ie_serving.server.predict_utils import statusCodes

logger = get_logger(__name__)


class Reshaper:

    def __init__(self, service_type):
        if service_type != GRPC and service_type != REST:
            raise ValueError("Provided service type is unavailable")
        self.service_type = service_type

    def detect_shapes_incompatibility(self, engine, inference_input):
        # Compares workload shapes with engine inputs shapes. If different,
        # returns True, reshape_param.
        # reshape_param is inputs shapes dictionary (input_name:shape pairs)
        # for reshapable models and batch size for non-reshapable

        reshape_required, inputs_shapes = engine.scan_input_shapes(
            inference_input)
        reshape_param = inputs_shapes
        # For non-reshapable models, batch_size of first input is the
        # reshape parameter
        if engine.shape_info.mode == ShapeMode.DISABLED:
            input_shape = inputs_shapes[list(inputs_shapes.keys())[0]]
            batch_size = list(input_shape)[0]
            reshape_param = batch_size
        return reshape_required, reshape_param

    def prepare_engine(self, engine, reshape_param, response_context):
        # Reshapes engine's inputs and changes response context on error.
        # Returns True if error occurred during reshaping.
        reshape_start_time = datetime.datetime.now()
        is_error, error_message = engine.reshape(reshape_param)
        reshape_end_time = datetime.datetime.now()
        if is_error:
            self._prepare_error_response(error_message, response_context)
            return True
        duration = \
            (reshape_end_time - reshape_start_time).total_seconds() * 1000
        logger.debug(
            "PREDICT; network reshape completed; {}; {}; {}ms".format(
                engine.model_name, engine.model_version, duration))
        return False

    def _prepare_error_response(self, error_message, response_context):
        # Changes codes and messages in response context. Does not return
        # any value.
        if self.service_type == GRPC:
            code = statusCodes['invalid_arg'][GRPC]
            response_context.set_code(code)
            response_context.set_details(error_message)
        elif self.service_type == REST:
            response_context.status = falcon.HTTP_400
            err_out_json = {'error': error_message}
            response_context.body = json.dumps(err_out_json)
