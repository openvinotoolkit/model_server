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

from ie_serving.logger import get_logger
from ie_serving.models.shape_management.utils import ShapeMode

logger = get_logger(__name__)


class ShapeInfo:

    def __init__(self, shape_param, net_inputs):
        shape_mode = ShapeMode.DISABLED
        shape = None
        if shape_param is not None:
            shape_mode, shape = self.process_shape_param(
                shape_param, net_inputs)
        self.mode = shape_mode
        self.shape = shape
        # shape field can be either:
        # - None for disabled, default, auto mode
        # - Dict of input_name:shape pairs for fixed mode

    def process_shape_param(self, shape_param, net_inputs):
        shape = None
        shape_mode = ShapeMode.DEFAULT
        if type(shape_param) is dict:
            shape = self.get_shape_dict(shape_param)
        elif type(shape_param) is str:
            shape_mode, shape = self.get_shape_from_string(shape_param)

        if shape is not None:
            shape_mode = ShapeMode.FIXED
            if type(shape) is tuple:
                shape = self._shape_as_dict(shape, net_inputs)

        return shape_mode, shape

    def _shape_as_dict(self, shape: tuple, net_inputs: dict):
        if len(net_inputs) > 1:
            raise Exception("Noname shape specified for model with "
                            "multiple inputs")
        else:
            input_name = list(net_inputs.keys())[0]
            return {input_name: shape}

    def get_shape_from_string(self, shape_param):
        shape = None
        shape_mode = ShapeMode.DEFAULT
        if shape_param == 'auto':
            shape_mode = ShapeMode.AUTO
            return shape_mode, shape
        shape_param = shape_param.replace('(', '[').replace(')', ']')
        shape = self.load_shape(shape_param)
        if shape is None:
            return shape_mode, shape

        if type(shape) is list:
            shape = self.get_shape_tuple(shape)
        elif type(shape) is dict:
            shape = self.get_shape_dict(shape)

        if shape is not None:
            shape_mode = ShapeMode.FIXED
        return shape_mode, shape

    def get_shape_dict(self, shapes: dict):
        output_shapes = {}
        for key, value in shapes.items():
            if type(key) is str and type(value) is str:
                value = value.replace('(', '[').replace(')', ']')
                output_shapes.update(self._get_single_shape(input_name=key,
                                                            shape=value))
        if output_shapes:
            return output_shapes
        return None

    def _get_single_shape(self, input_name, shape):
        shape = self.load_shape(shape)
        if shape is not None and type(shape) is list:
            shape = self.get_shape_tuple(shape)
            if shape is not None:
                return {input_name: shape}
        return {}

    def get_shape_tuple(self, shape: list):
        try:
            shape = tuple([int(dim) for dim in shape])
        except Exception as e:
            logger.error("Error getting shape tuple: {}"
                         .format(str(e)))
            shape = None
        return shape

    def load_shape(self, shape_param: str):
        shape = None
        try:
            shape = json.loads(shape_param)
        except Exception as e:
            logger.error("Error getting shapes from the string: {}"
                         .format(str(e)))
        return shape
