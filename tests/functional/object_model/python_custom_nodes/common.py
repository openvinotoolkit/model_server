#
# Copyright (c) 2026 Intel Corporation
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

from pathlib import Path

from tests.functional.config import ovms_c_repo_path

PYTHON_CUSTOM_NODE_EXPECTED_CLASS_NAME = "OvmsPythonModel"
PYTHON_CUSTOM_NODES_DIR = Path(ovms_c_repo_path, "tests", "functional", "data", "python_custom_nodes")


class OvmsPythonModelFiles:
    PYTHON_MODEL_FILEPATH = Path(PYTHON_CUSTOM_NODES_DIR, "ovms_basic/python_model.py")
    PYTHON_MODEL_LOOPBACK_FILEPATH = Path(PYTHON_CUSTOM_NODES_DIR, "ovms_basic/python_model_loopback.py")
    # corrupted model.py files
    PYTHON_MODEL_EXCEPTIONS_FILEPATH = Path(PYTHON_CUSTOM_NODES_DIR, "ovms_corrupted/python_model_exceptions.py")
    PYTHON_MODEL_CORRUPTED_IMPORT_FILEPATH = Path(
        PYTHON_CUSTOM_NODES_DIR, "ovms_corrupted/python_model_corrupted_import.py"
    )
    PYTHON_MODEL_MISSING_EXECUTE_FILEPATH = Path(
        PYTHON_CUSTOM_NODES_DIR, "ovms_corrupted/python_model_missing_execute.py"
    )
    PYTHON_INCREMENTER_FILEPATH = Path(PYTHON_CUSTOM_NODES_DIR, "incrementer/incrementer.py")
    PYTHON_MODEL_LOOPBACK_RETURN_INSTEAD_OF_YIELD_FILEPATH = Path(
        PYTHON_CUSTOM_NODES_DIR, "ovms_corrupted/python_model_loopback_return_instead_of_yield.py"
    )
    PYTHON_MODEL_WRITING_TO_LOOPBACK_OUTPUT_IN_EXECUTE_FILEPATH = Path(
        PYTHON_CUSTOM_NODES_DIR, "ovms_corrupted/python_model_writing_to_loopback_output_in_execute.py"
    )
    PYTHON_MODEL_MULTIPLE_USE_OF_VALID_OUTPUTS_FILEPATH = Path(
        PYTHON_CUSTOM_NODES_DIR, "ovms_corrupted/python_model_loopback_multiple_use_of_valid_outputs.py"
    )


STREAMING_CHANNEL_ARGS = [
    # Do not drop the connection for long workloads
    ("grpc.http2.max_pings_without_data", 0),
]
