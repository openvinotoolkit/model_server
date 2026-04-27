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

import signal
import sys

import pytest

from tests.functional.utils.assertions import (
    OvmsTestException,
    UnexpectedResponseError,
)
from tests.functional.utils.context import Context
from tests.functional.utils.logger import get_logger, log_fixture
from tests.functional.utils.test_framework import skip_if_runtime

from tests.functional.config import (
    resource_monitor_enabled,
    run_ovms_with_opencl_trace,
    run_ovms_with_valgrind,
)
from tests.functional.constants.ovms_binaries import get_binaries
from tests.functional.constants.ovms_type import OvmsType
from tests.functional.object_model.tools import Cliloader, Valgrind
from tests.functional.object_model.ovms_binary import start_binary_ovms
from tests.functional.object_model.ovms_capi import OvmsCapiParams, start_capi_ovms
from tests.functional.object_model.ovms_docker import OvmsDockerLauncher, OvmsDockerParams
from tests.functional.object_model.ovms_instance import OvmsRunContext
from tests.functional.object_model.test_environment import TestEnvironment

logger = get_logger(__name__)


class SigtermCleaner:
    def __init__(self):
        self.test_objects = []
        signal.signal(signal.SIGTERM, self.sigterm_handle)

    def sigterm_handle(self, _signo, _stack_frame):
        logger.warning(f"Received signal={_signo} from:\n{_stack_frame}")
        while len(self.test_objects) > 0:
            item = self.test_objects.pop()
            name = getattr(item, "name", str(type(item)))
            logger.warning(f"Cleanup test object: {name}")
            try:
                item.cleanup()
            except (UnexpectedResponseError, AssertionError) as exc:
                logger.exception(str(exc))
                pass
        sys.exit(1)


@pytest.fixture(scope="session")
def sigterm_cleaner():
    return SigtermCleaner()


def start_ovms(
    context: Context,
    parameters,
    environment: dict = None,
    ensure_started: bool = True,
    entrypoint=None,
    entrypoint_params=None,
    ovms_type_to_start=None,
    ensure_nodeport: bool = True,
    ovms_instance_params=None,
    valgrind_mode=None,
    terminate_signal_type=None,
    terminate_method=None,
    timeout=None,
    **kwargs,
):
    if ovms_type_to_start is None:
        ovms_type_to_start = context.ovms_type

    context.terminate_signal_type = terminate_signal_type
    context.terminate_method = terminate_method

    if run_ovms_with_valgrind or valgrind_mode is not None:
        entrypoint = Valgrind.name
        valgrind_mode = "basic" if valgrind_mode is None else valgrind_mode
        entrypoint_params = Valgrind.get_valgrind_params(valgrind_mode)
    if run_ovms_with_opencl_trace:
        entrypoint = Cliloader.path
        env = Cliloader.env
        if environment is not None:
            environment.update(env)
        else:
            environment = env

    if ovms_type_to_start in (OvmsType.DOCKER, OvmsType.DOCKER_CMD_LINE, OvmsType.BINARY_DOCKER):
        ovms_docker_params = rewrite_parameters(inherit_class=OvmsDockerParams, base_class_object=parameters)
        result = start_docker_ovms(
            context,
            ovms_docker_params,
            environment,
            ovms_type_to_start,
            ovms_instance_params,
            entrypoint,
            entrypoint_params,
        )
    elif ovms_type_to_start == OvmsType.BINARY:
        ovms_binary_path = kwargs.get("ovms_binary_path", None)
        ovms_binary_name = kwargs.get("ovms_binary_name", None)
        if ovms_binary_path is None or ovms_binary_name is None:
            ovms_binary_path, ovms_binary_name = get_binaries(
                context.base_os,
                context.test_object_name,
                TestEnvironment.current.base_dir,
            )
        parameters.name = parameters.name if parameters.name is not None else ovms_binary_name
        result = start_binary_ovms(context, parameters, ovms_binary_path, environment, **kwargs)
    elif ovms_type_to_start == OvmsType.CAPI:
        ovms_capi_params = rewrite_parameters(inherit_class=OvmsCapiParams, base_class_object=parameters)
        result = start_capi_ovms(context=context, parameters=ovms_capi_params, environment=environment)
    elif ovms_type_to_start == OvmsType.NONE:
        logger.warning("SKIP: Executing Ovms tests with ovms_type == 'NONE'")
        skip_if_runtime(True, msg="'NONE' type not supported")
    else:
        raise OvmsTestException(f"Unrecognized ovms_type_to_start={ovms_type_to_start}")

    result.attach_context(context)

    result.ovms._dmesg_log.ovms_pid = result.ovms.fetch_and_store_ovms_pid()

    if ensure_started:
        assert not parameters.check_version, "OVMS container will not start if --version argument was given."
        log_fixture(
            "Ensure ovms is running with model(s): {}".format(", ".join([model.name for model in result.models]))
        )
        result.ovms.ensure_started(result.models, timeout=timeout, os_type=context.base_os)

    return result


def rewrite_parameters(inherit_class, base_class_object):
    inherit_object = inherit_class()
    for key, value in base_class_object.__dict__.items():
        if base_class_object.__dict__[key] is not None:
            inherit_object.__dict__[key] = value
    return inherit_object


def start_docker_ovms(
    context: Context,
    parameters,
    environment: dict,
    ovms_docker_type,
    ovms_instance_params=None,
    entrypoint=None,
    entrypoint_params=None,
) -> OvmsRunContext:
    if getattr(context, "ovms_test_image", None) is not None and context.ovms_type != OvmsType.BINARY_DOCKER:
        # If testing image was build & set in context:
        # replace default image with testing image. In near future all tests should use `ovms_test_image`
        parameters.image = context.ovms_test_image

    if parameters.ports_enabled():
        if context.port_manager_grpc is not None and parameters.grpc_port is None:
            parameters.grpc_port = context.port_manager_grpc.get_port()
        if context.port_manager_rest is not None and parameters.rest_port is None:
            parameters.rest_port = context.port_manager_rest.get_port()

    ovms_instance = OvmsDockerLauncher.create(
        context, parameters, ovms_docker_type, environment, entrypoint, entrypoint_params, ovms_instance_params
    )
    ovms_run_context = OvmsRunContext(ovms_instance, parameters.models)
    if resource_monitor_enabled:
        ovms_run_context.attach_resource_monitor(context)
    return ovms_run_context
