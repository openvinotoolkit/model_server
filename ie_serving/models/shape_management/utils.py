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

    def __init__(self, shape_mode):
        self.mode = shape_mode

    @classmethod
    def build(cls, shape_param):
        if shape_param is not None:
            if shape_param == 'auto':
                shape_mode = ShapeMode.AUTO
            else:
                shape_mode = ShapeMode.DEFAULT
        else:
            shape_mode = ShapeMode.DISABLED
        return cls(shape_mode)
