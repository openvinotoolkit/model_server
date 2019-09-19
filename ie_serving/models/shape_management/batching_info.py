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
from ie_serving.logger import get_logger
from ie_serving.models.shape_management.utils import BatchingMode

logger = get_logger(__name__)


class BatchingInfo:
    def __init__(self, batch_size_param):
        self.batch_size = None
        self.mode = BatchingMode.DEFAULT
        if batch_size_param is not None:
            if batch_size_param.isdigit() and int(batch_size_param) > 0:
                self.mode = BatchingMode.FIXED
                self.batch_size = int(batch_size_param)
            elif batch_size_param == 'auto':
                self.mode = BatchingMode.AUTO

    def get_effective_batch_size(self):
        if self.mode == BatchingMode.AUTO:
            return "auto"
        if self.batch_size is not None:
            return str(self.batch_size)
