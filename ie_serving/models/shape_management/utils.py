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

logger = get_logger(__name__)
##############################################
# Input batch size and shape auxiliary classes


class GenericMode:
    FIXED = 0
    AUTO = 1
    DEFAULT = 2
    DISABLED = 3


class BatchingMode(GenericMode):
    pass


class ShapeMode(GenericMode):
    pass


class BatchingInfo:
    def __init__(self, batch_size_mode, batch_size):
        self.mode = batch_size_mode
        self.batch_size = batch_size

    @classmethod
    def build(cls, batch_size_param):
        batch_size = None
        batch_size_mode = BatchingMode.DEFAULT
        if batch_size_param is not None:
            if batch_size_param.isdigit() and int(batch_size_param) > 0:
                batch_size_mode = BatchingMode.FIXED
                batch_size = int(batch_size_param)
            elif batch_size_param == 'auto':
                batch_size_mode = BatchingMode.AUTO
            else:
                batch_size_mode = BatchingMode.DEFAULT
        return cls(batch_size_mode, batch_size)

    def get_effective_batch_size(self):
        if self.mode == BatchingMode.AUTO:
            return "auto"
        if self.batch_size is not None:
            return str(self.batch_size)


class ShapeInfo:

    def __init__(self, shape_mode, shape):
        self.mode = shape_mode
        self.shape = shape
        # shape field can be either:
        # - None for disabled, default, auto mode
        # - Tuple or Dict in fixed mode, depending on provided parameter

    def get_shape_dict(self, model_inputs):
        # Returns shape in dict format - {input_name: shape, ...}
        if type(self.shape) is tuple:
            if len(list(model_inputs.keys())) > 1:
                raise Exception("Noname shape specified for model with "
                                "multiple inputs")
            else:
                input_name = list(model_inputs.keys())[0]
                fixed_shape = {input_name: self.shape}
        elif type(self.shape) is dict:
            fixed_shape = self.shape
        else:
            raise Exception("Unexpected fixed shape type")
        return fixed_shape

    @classmethod
    def build(cls, shape_param):
        if shape_param is None:
            return cls(ShapeMode.DISABLED, None)
        shape_mode, shape = cls.process_shape_param(shape_param)
        return cls(shape_mode, shape)

    @classmethod
    def process_shape_param(cls, shape_param):
        shape = None
        shape_mode = ShapeMode.DEFAULT
        if type(shape_param) is dict:
            shape = cls.prepare_shape_dict(shape_param)
            shape_mode = ShapeMode.FIXED
        elif type(shape_param) is str:
            shape_mode, shape = cls.get_shape_from_string(shape_param)
        return shape_mode, shape

    @classmethod
    def get_shape_from_string(cls, shape_param):
        shape = None
        shape_mode = ShapeMode.DEFAULT
        if shape_param == 'auto':
            shape_mode = ShapeMode.AUTO
        elif shape_param[0] == '(' and shape_param[-1] == ')':
            shape = cls.parse_shape_tuple(shape_param)
            if shape is not None:
                shape_mode = ShapeMode.FIXED
        elif shape_param[0] == '{' and shape_param[-1] == '}':
            shape = cls.parse_shape_dict(shape_param)
            if shape:
                shape_mode = ShapeMode.FIXED
        logger.warning('Unexpected value in shape parameter. Using default.')
        return shape_mode, shape

    @classmethod
    def parse_shape_dict(cls, shape_param: str):
        try:
            shapes_dict = json.loads(shape_param)
        except Exception as e:
            logger.error("Error getting shapes dictionary from string: {}"
                         .format(str(e)))
            return {}
        return cls.prepare_shape_dict(shapes_dict)

    @classmethod
    def prepare_shape_dict(cls, shapes_dict):
        output_shapes = {}
        for key, value in shapes_dict.items():
            if type(key) is str and type(value) is str:
                shape = cls.parse_shape_tuple(value)
                if shape is not None:
                    output_shapes[key] = shape
        return output_shapes

    @classmethod
    def parse_shape_tuple(cls, shape_param: str):
        shape_param = shape_param.strip()
        shape_str = shape_param[1:-1].split(',')
        try:
            shape = tuple([int(shape.strip()) for shape in shape_str])
        except Exception as e:
            logger.error("Error getting single shape tuple from string: {}"
                         .format(str(e)))
            shape = None
        return shape
