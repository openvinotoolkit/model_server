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
import pickle
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from select import select
from typing import List

import numpy as np

from tests.functional.utils.assertions import CapiException
from tests.functional.utils.context import Context
from tests.functional.utils.logger import get_logger
from tests.functional.utils.test_framework import generate_test_object_name, skip_if_runtime
from tests.functional.config import ovms_c_repo_path
from tests.functional.constants.core import CONTAINER_STATUS_RUNNING
from tests.functional.constants.models import ModelInfo
from tests.functional.constants.ovms import Config
from tests.functional.constants.ovms_messages import OvmsMessages
from tests.functional.constants.ovms_type import OvmsType
from tests.functional.constants.paths import Paths
from tests.functional.utils.log_monitor import LogMonitor
from tests.functional.object_model.cpu_extension import MuseModelExtension
from tests.functional.object_model.ovms_binary import OvmsBinary
from tests.functional.object_model.ovms_config import OvmsConfig
from tests.functional.object_model.ovms_docker import OvmsDockerParams
from tests.functional.object_model.ovms_instance import OvmsRunContext
from tests.functional.object_model.ovms_log_monitor import BinaryOvmsLogMonitor
from tests.functional.object_model.ovms_params import OvmsParams
from tests.functional.object_model.test_environment import TestEnvironment
from tests.functional.utils.remote_test_environment import copy_custom_lib_to_host

logger = get_logger(__name__)


@dataclass(frozen=False)
class OvmsCapiParams(OvmsParams):
    config_path: str = ""
    cpu_extension_path: str = ""


class OvmsCapiLogMonitor(BinaryOvmsLogMonitor):
    def is_running(self, ovms_ports, timeout, os_type=None):
        return True


@dataclass
class OvmsCapiRunContext(OvmsRunContext):
    addr: int = 0


class OvmsCapiInstance(OvmsBinary):
    PYTHON_SNIPPET = "'from tests.functional.object_model.ovms_capi import OvmsCapiInstance; OvmsCapiInstance({},{}).main_loop({})'"

    def __init__(self, parameters, base_os, **kwargs):
        if isinstance(parameters, dict):
            # spawned instance
            logger.info(parameters)
            parameters = OvmsCapiParams(**parameters)
        self.base_os = base_os
        self.parameters = parameters
        self.pickle_friendly_parameters = self.parameters.to_str()
        capi_package_content = Paths.CAPI_WRAPPER_PACKAGE_CONTENT_PATH(self.base_os)
        cmd = self.PYTHON_SNIPPET.format(
            f"parameters={self.pickle_friendly_parameters}",
            f'base_os="{self.base_os}"',
            f'capi_package_content="{capi_package_content}"',
        )

        super().__init__(
            name=parameters.name,
            cmd=cmd,
            parameters=parameters,
            path_to_binary=None,
            container_folder=getattr(parameters, "resources_dir", None),
            **kwargs,
        )

    def get_port(self, api_type):
        if not isinstance(api_type, str) and api_type.communication == OvmsType.CAPI:
            return self
        else:
            return super().get_port(api_type)

    def get_status(self, status=None, timeout=None):
        status = Path(f"/proc/{self.process._proc.pid}/status").read_text()
        status = [line for line in status.splitlines() if line.startswith("State:")]
        if "sleeping" in status[0]:
            result = CONTAINER_STATUS_RUNNING
        else:
            result = None
        return result

    def start(self, ensure_started=False, use_valgrind=False, *args, **kwargs):
        cmd = [sys.executable, "-c", f"{self.cmd}", "2>&1"]
        if use_valgrind:
            cmd = cmd.insert(0, "valgrind")
        cmd = " ".join(cmd)

        self.process.async_run(cmd, cwd=os.getcwd(), daemon_mode=True, use_stdin=True)
        self.ovms_pid = self.process._proc.pid
        self._stdin_proc_fd = f"/proc/{self.ovms_pid}/fd/0"

    def cleanup(self):
        try:
            self.send_terminate_command()
        except Exception as e:
            logger.error(e)
        time.sleep(1)  # give some time for terminate
        logger.debug("Cleaning OVMS CAPI instance")
        super().cleanup()

    def _create_logger(self) -> LogMonitor:
        return OvmsCapiLogMonitor(self.process)

    def execute_command(self, cmd, cwd=None):
        # Execute command from host.
        try:
            # Execute python api command without any validation (sic!).
            # Expect valid command sent by host process.
            print(cmd)
            result = exec(cmd)
        except Exception as e:
            logger.exception(e)

    def ensure_started(
        self,
        expected_loaded_models: List[ModelInfo] = None,
        custom_messages: list = None,
        expected_unloaded_models: List[ModelInfo] = None,
        timeout: int = None,
        os_type: str = None,
    ):
        if not custom_messages:
            custom_messages = []
        custom_messages.append(OvmsMessages.CAPI_STARTED_OVMS_SERVER)
        self._default_log.ensure_contains_messages(custom_messages, timeout=timeout, ovms_instance=self)

    @staticmethod
    def ensure_newline(cmd):
        return cmd if cmd.endswith("\n") else f"{cmd}\n"

    def send_command_to_process(self, cmd):
        # Write directly to stdin file descriptor in /proc/<pid> filesystem.
        # It should mitigate interprocess communication issues in 'pure pythonic' approach.
        logger.debug(self._stdin_proc_fd)
        with open(self._stdin_proc_fd, "w") as fd:
            fd.write(self.ensure_newline(cmd))

    def send_command_to_process_with_output(self, cmd, cmd_kwargs):
        with tempfile.NamedTemporaryFile() as tmp_infile:
            with tempfile.NamedTemporaryFile() as tmp_outfile:
                cmd = f"self.capi.tmp_infile='{tmp_infile.name}'; self.capi.tmp_outfile='{tmp_outfile.name}'; {cmd}"
                Path(tmp_infile.name).write_bytes(pickle.dumps(cmd_kwargs))

                self.send_command_to_process(cmd)
                output_path = Path(tmp_outfile.name)

                t_end = time.time() + 20
                while time.time() < t_end and output_path.stat().st_size == 0:
                    time.sleep(0.5)
                assert output_path.stat().st_size
                raw_data = output_path.read_bytes()
                result = pickle.loads(raw_data)
        if issubclass(type(result), Exception):
            raise result
        return result

    def send_start_server_command(self):
        cmd = f"self.srv = self.capi.ovms_start_server({self.pickle_friendly_parameters})"
        return self.send_command_to_process(cmd)

    def send_stop_server_command(self):
        cmd = f"self.srv = self.capi.server_stop()"
        return self.send_command_to_process(cmd)

    def send_terminate_command(self):
        cmd = f"self.running = False"
        return self.send_command_to_process(cmd)

    def send_terminate_command(self):
        cmd = f"self.running = False"
        return self.send_command_to_process(cmd)

    def send_get_model_meta_command(self, model_name, model_version):
        cmd = f"self.capi.get_model_meta()"
        cmd_kwargs = {"servableName": model_name, "servableVersion": model_version}
        return self.send_command_to_process_with_output(cmd, cmd_kwargs)

    def send_inference(self, model, input_data):
        if not input_data:
            skip_if_runtime(
                any(-1 in shape for input_name, shape in model.input_shapes.items()),
                msg="Dynamic shapes not supported for now",
            )

        _in_data = {key: value.shape for key, value in input_data.items()}
        # Write directly to stdin file descriptor in /proc/<pid> filesystem.
        # It should mitigate interprocess communication issues in 'pure pythonic' approach.
        cmd = f"self.result = self.capi.send_inference()"
        cmd_kwargs = {
            "model_name": model.name,
            "inputs": input_data,
        }
        result = self.send_command_to_process_with_output(cmd, cmd_kwargs)
        if "Unsupported data type" in result:
            raise CapiException(result)
        result = {key: np.asarray(val) for key, val in result.items()}
        return result

    def send_get_capi_api_version(self):
        cmd = "self.major, self.minor = self.capi.ovms_get_capi_version()"
        self.send_command_to_process(cmd)

    def get_major_minor_version(self):
        filepath = os.path.join(ovms_c_repo_path, "src/ovms.h")
        with open(filepath, "r") as f:
            data = f.read()

        major = re.search(r"OVMS_API_VERSION_MAJOR (\d+)", data).group(1)
        minor = re.search(r"OVMS_API_VERSION_MINOR (\d+)", data).group(1)

        return major, minor

    def fetch_command(self):
        # Expect commands submitted by host process via OvmsCapiInstance.send_command_to_process
        cmd = input()  # Simple blocking read should suffice for now.
        return cmd.strip()

    def main_loop(self, capi_package_content):
        """
        This method should be called in spawned thread.
        """
        # Prepare PYTHONPATHs prior imports
        sys.path.append(os.path.join(capi_package_content, "lib"))

        import ovms_capi_wrapper as capi  # pylint: disable=import-outside-toplevel, import-error

        self.capi = capi
        self.running = True
        while self.running:
            rlist, _, _ = select([sys.stdin], [], [], 1.0)
            if rlist:
                cmd = self.fetch_command()
                self.execute_command(cmd)
            else:
                print("Sleeping ...")
                time.sleep(0.5)


def start_capi_ovms(context: Context, parameters: OvmsDockerParams, environment: dict = None, resources_dir=None):
    if parameters.name is None:
        parameters.name = (
            context.test_object_name if context.test_object_name is not None else generate_test_object_name()
        )
    if not resources_dir:
        parameters.resources_dir, _ = TestEnvironment.current.prepare_container_folders(
            parameters.name, parameters.get_models()
        )
    else:
        parameters.resources_dir = resources_dir

    if parameters.cpu_extension:
        if isinstance(parameters.cpu_extension, MuseModelExtension):
            capi_package = Path(Paths.CAPI_WRAPPER_PACKAGE_CONTENT_PATH(context.base_os))
            parameters.cpu_extension_path = str(Path(capi_package.parent, f"./{parameters.cpu_extension.lib_path}"))
        else:
            host_dir = os.path.join(parameters.resources_dir, Paths.CPU_EXTENSIONS)
            host_lib_path = os.path.join(host_dir, parameters.cpu_extension.lib_name)
            copy_custom_lib_to_host(context.ovms_test_image, parameters.cpu_extension.lib_path, host_lib_path)
            parameters.cpu_extension_path = str(host_lib_path)

    if context.port_manager_grpc is not None and parameters.grpc_port is None:
        parameters.grpc_port = context.port_manager_grpc.get_port()
    if context.port_manager_rest is not None and parameters.rest_port is None:
        parameters.rest_port = context.port_manager_rest.get_port()

    # https://github.com/openvinotoolkit/model_server/blob/main/src/ovms.h#L394
    # CAPI is supported only with config.json file.
    if parameters.custom_config is None:
        indirect_config_path, config_dict = OvmsConfig.generate_from_parameters(
            parameters.name, parameters, parameters.resources_dir
        )
    else:
        config_dict = parameters.custom_config
        for model in config_dict[Config.MODEL_CONFIG_LIST]:
            if not model["config"]["base_path"].startswith(parameters.resources_dir):
                model_config_base_path = model["config"]["base_path"]
                model["config"]["base_path"] = os.path.join(parameters.resources_dir, model_config_base_path.strip("/"))
        indirect_config_path = OvmsConfig.save(parameters.name, config_dict)
    parameters.config_path = str(Path(parameters.resources_dir, f"./{indirect_config_path}"))

    capi_instance = OvmsCapiInstance(parameters, context.base_os)
    capi_instance.start()  # Execute main loop
    context.test_objects.append(capi_instance)

    capi_instance.send_start_server_command()

    return OvmsCapiRunContext(capi_instance, parameters.models)
