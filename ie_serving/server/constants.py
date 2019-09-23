#
# Copyright (c) 2018-2019 Intel Corporation
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

WRONG_MODEL_SPEC = 'Servable not found for request: Specific({}, {})'
INVALID_INPUT_KEY = 'input tensor alias not found in signature: %s. ' \
                    'Inputs expected to be in the set {%s}.'
INVALID_SHAPE = 'The input data is incorrect. Obtained shape {}, ' \
                'required shape {}'
INVALID_METADATA_FIELD = 'Metadata field {} is not supported'

INVALID_BATCHSIZE = 'Input batch size is incorrect. Obtained batch size {}, '\
                    'required batch size {}'
CONFLICTING_PARAMS_WARNING = "Both shape and batch_size parameters " \
                              "are set for model: {}. Assuming that model is"\
                              " reshapable - batch_size will be ignored"

SIGNATURE_NAME = "serving_default"

INVALID_FORMAT, ROW_FORMAT, ROW_SIMPLIFIED, COLUMN_FORMAT, COLUMN_SIMPLIFIED \
    = 0, 1, 2, 3, 4

OUTPUT_REPRESENTATION = {
    ROW_FORMAT: 'row',
    ROW_SIMPLIFIED: 'row',
    COLUMN_FORMAT: 'column',
    COLUMN_SIMPLIFIED: 'column'
}

GRPC = 0
REST = 1
