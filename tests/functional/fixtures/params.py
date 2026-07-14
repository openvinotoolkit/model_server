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

import os
import pytest

from tests.functional.utils.logger import get_logger
from tests.functional.utils.test_framework import create_venv_and_install_packages

from tests.functional.config import tmp_dir
from tests.functional.constants.ovms import MediapipeIntermediatePacket, Ovms
from tests.functional.constants.paths import Paths

logger = get_logger(__name__)


@pytest.fixture(scope="session", params=[True, False], ids=["delete_enable_file", "erase_enable_file"])
def delete_enable_file(request):
    return request.param


@pytest.fixture(scope="session", params=[Ovms.JPG_IMAGE_FORMAT, Ovms.PNG_IMAGE_FORMAT])
def image_format(request):
    return request.param


@pytest.fixture(scope="session", params=[Ovms.LAYOUT_NHWC, Ovms.LAYOUT_NCHW])
def layout(request):
    return request.param


@pytest.fixture(scope="function", params=[True, False], ids=["with_config", "without_config"])
def use_config(request):
    return request.param


@pytest.fixture(scope="function", params=[True, False], ids=["with_subconfig", "without_subconfig"])
def use_subconfig(request):
    return request.param


@pytest.fixture(scope="function", params=[True, False], ids=["relative_paths", "absolute_paths"])
def use_relative_paths(request):
    return request.param


@pytest.fixture(scope="function", params=["basic", "full"])
def valgrind_mode(request):
    return request.param


@pytest.fixture(scope="function", params=Ovms.V2_OPERATIONS)
def operation(request):
    return request.param


@pytest.fixture(scope="function", params=[x.name for x in list(MediapipeIntermediatePacket)])
def mediapipe_intermediate_type_graph(request):
    return request.param


@pytest.fixture(scope="session")
def optimum_cli_activate_path():
    # prepare venv with optimum-cli needed for downloading models that require conversion
    venv_activate_path = create_venv_and_install_packages(
        os.path.join(tmp_dir, "optimum_cli_requirements"),
        requirements_file_path=Paths.LLM_EXPORT_MODELS_REQUIREMENTS,
    )
    return venv_activate_path
