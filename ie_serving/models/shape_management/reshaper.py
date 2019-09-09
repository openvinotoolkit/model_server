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
import datetime

from ie_serving.logger import get_logger
from ie_serving.models.shape_management.utils import ShapeMode


logger = get_logger(__name__)


class Reshaper:

    @staticmethod
    def detect_shapes_incompatibility(engine, inference_input):
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

    @staticmethod
    def prepare_engine(engine, reshape_param):
        # Reshapes engine's inputs and changes response context on error.
        # Returns True if error occurred during reshaping.
        reshape_start_time = datetime.datetime.now()
        error_message = engine.reshape(reshape_param)
        reshape_end_time = datetime.datetime.now()
        if error_message is not None:
            return error_message
        duration = \
            (reshape_end_time - reshape_start_time).total_seconds() * 1000
        logger.debug(
            "RESHAPER; network reshape completed; {}; {}; {}ms".format(
                engine.model_name, engine.model_version, duration))
        return None
