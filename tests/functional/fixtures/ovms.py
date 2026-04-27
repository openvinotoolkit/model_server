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

import platform
import time
from typing import Callable

import pytest
from _pytest.fixtures import FixtureRequest

from tests.functional.utils.context import Context
from tests.functional.utils.environment_info import EnvironmentInfo
from tests.functional.utils.inference.communication import GRPC, REST
from tests.functional.utils.logger import get_logger
from tests.functional.utils.marks import MarkGeneral
from tests.functional.constants.os_type import OsType
from tests.functional.utils.port_manager import PortManager
from tests.functional.utils.test_framework import generate_test_object_name, skip_if_runtime

from tests.functional.config import (
    build_test_image,
    delay_between_test,
    grpc_ovms_starting_port,
    ports_pool_size,
    pytest_global_session_timeout,
    rest_ovms_starting_port,
    run_ovms_with_opencl_trace,
    run_ovms_with_valgrind,
)
from tests.functional.constants.models import ModelInfo
from tests.functional.constants.ovms import CurrentOvmsType, CurrentTarget
from tests.functional.constants.ovms_binaries import calculate_ovms_binary_name
from tests.functional.constants.ovms_images import (
    calculate_ovms_binary_image_name,
    calculate_ovms_image_name,
    calculate_ovms_test_image_name,
)
from tests.functional.constants.ovms_type import OvmsType
from tests.functional.constants.requirements import Requirements
from tests.functional.object_model.cpu_extension import CpuExtension
from tests.functional.object_model.test_environment import TestEnvironment
from tests.functional.object_model.ovms_info import OvmsInfo

logger = get_logger(__name__)


@pytest.fixture(scope="function")
def add_test_finalizer(request: FixtureRequest) -> Callable[..., None]:
    return request.addfinalizer


@pytest.fixture(scope="function", autouse=True)
def context(request, sigterm_cleaner, target_device, ovms_type, base_os):
    context = Context(request.scope, request.node.nodeid, request.node)
    request.addfinalizer(context.cleanup)
    sigterm_cleaner.test_objects.append(context)

    context.target_device = target_device
    context.ovms_type = ovms_type
    context.base_os = base_os
    logger.info(f"Running test on platform: {platform.node()}")
    if base_os == OsType.Windows:
        context.ovms_image = None
    else:
        context.ovms_image = calculate_ovms_image_name(context.target_device, context.base_os)
        if ovms_type == OvmsType.BINARY_DOCKER:
            context.ovms_image = calculate_ovms_binary_image_name(context.ovms_image)
    if OvmsType.DOCKER not in ovms_type:
        context.ovms_binary = calculate_ovms_binary_name(context.base_os)
    context.ovms_sessions = []

    # Check if tests is marked by @pytest.mark.reqids(...) with libs build in ovms-testing-image
    reqids_node = [x for x in request.node.own_markers if x.name == MarkGeneral.REQIDS.value]
    reqids_parent = [x for x in request.node.parent.own_markers if x.name == MarkGeneral.REQIDS.value]
    requirements_with_external_libraries = [
        Requirements.custom_loader,
        Requirements.cpu_extension,
        Requirements.custom_nodes,
        Requirements.valgrind,
    ]
    classes_with_external_libraries_used = ["TestByXCli2"]
    use_ovms_testing_image = any([
        reqids_node
        and any([x in requirements_with_external_libraries for x in reqids_node[0].args]),  # By requirement id
        reqids_parent and any([x in requirements_with_external_libraries for x in reqids_parent[0].args]),
        request.node.parent.name in classes_with_external_libraries_used,
    ])
    # Currently we enable testing image only for test that require custom build binaries:
    # (custom nodes, cpu_extensions, etc.). In near future we wish to use only ovms-testing-image.
    if use_ovms_testing_image or run_ovms_with_valgrind or run_ovms_with_opencl_trace:
        if not build_test_image:
            logger.warning(f"Skipping test {request.node.name} because lack of built ovms-testing-image")
            skip_if_runtime(True, msg="ovms-testing-image was not built")
        context.ovms_test_image = calculate_ovms_test_image_name(context.ovms_image)

    if request.cls.__name__ == "TestStraceOvmsMonitor":
        context.ovms_image = f"{context.ovms_image}-strace"

    context.env_info = EnvironmentInfo.get_instance(class_info=OvmsInfo, image=context.ovms_image)

    CpuExtension.base_os = context.base_os
    ModelInfo.base_os = context.base_os
    ModelInfo.target_device = context.target_device
    # Setup helper class for is_*_target
    CurrentTarget.target_device = context.target_device
    # Setup helper class for is_*_type
    CurrentOvmsType.ovms_type = context.ovms_type

    context.port_manager_grpc = PortManager(
        GRPC, starting_port=grpc_ovms_starting_port, pool_size=ports_pool_size
    )
    context.port_manager_rest = PortManager(
        REST, starting_port=rest_ovms_starting_port, pool_size=ports_pool_size
    )

    context.test_object_name = generate_test_object_name()
    return context


@pytest.fixture(scope="function", autouse=bool(delay_between_test))
def pause_after_test():
    yield
    time.sleep(delay_between_test)


@pytest.fixture(scope="function", autouse=True)
def set_test_environment(tmpdir):
    TestEnvironment.current = TestEnvironment(tmpdir)


@pytest.fixture(autouse=True)
def check_session_time(request):
    elapsed = time.time() - request.session.start_time
    if elapsed > (pytest_global_session_timeout * 60 * 60):
        msg = f"Pytest exited due to session timeout: {elapsed}s elapsed"
        logger.error(msg)
        pytest.exit(msg, returncode=-1)
