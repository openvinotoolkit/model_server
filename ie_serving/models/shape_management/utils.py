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
