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
import random
import re
import shutil
import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import sleep
from typing import List

from docker.errors import APIError

import tests.functional.utils.assertions as assertions_module
from tests.functional.utils.assertions import (
    CPP_STD_EXCEPTION,
    DmesgError,
    DockerCannotCloseProperly,
    OvmsTestException,
    UnwantedMessageError,
    get_exception_by_ovms_log,
)
from tests.functional.utils.context import Context
from tests.functional.utils.core import SelfDeletingFileLock
from tests.functional.utils.core import get_children_from_module
from tests.functional.utils.inference.communication import GRPC, REST
from tests.functional.utils.logger import get_logger
from tests.functional.constants.os_type import OsType
from tests.functional.utils.port_manager import PortManager
from tests.functional.utils.process import Process
from tests.functional.utils.test_framework import change_dir_permissions, is_single_threaded
from ovms.config import artifacts_dir, container_proxy, disable_dmesg_log_monitor, machine_is_reserved_for_test_session, \
    wait_for_messages_timeout
from tests.functional.constants.core import CONTAINER_STATUS_EXITED, CONTAINER_STATUS_RUNNING
from tests.functional.constants.models import ModelInfo
from tests.functional.constants.target_device import MAX_WORKERS_PER_TARGET_DEVICE
from tests.functional.constants.ovms import CurrentTarget as ct
from tests.functional.constants.ovms import Ovms
from tests.functional.constants.ovms_messages import OvmsMessages
from tests.functional.constants.paths import Paths
from tests.functional.constants.pipelines import Pipeline
from tests.functional.utils.log_monitor import LogMonitor
from tests.functional.object_model.dmesg_log_monitor import DmesgLogMonitor, DummyLogMonitor
from tests.functional.object_model.mediapipe_calculators import MediaPipeCalculator
from tests.functional.object_model.ovms_config import OvmsConfig
from tests.functional.object_model.package_manager import PackageManager
from tests.functional.object_model.resource_monitor import DockerResourceMonitor
from tests.functional.object_model.test_environment import TestEnvironment

logger = get_logger(__name__)


class OvmsInstance(ABC):

    def __init__(
        self,
        name,
        container_folder,
        default_logger,
        rest_port,
        grpc_port,
        target_device=None,
        remote_ip=None,
        lock_file=None,
        context=None,
        full_command=None,
    ):
        self.name = name
        self.container_folder = container_folder
        self.context = context
        self.ovms_pid = None
        self._default_log = default_logger
        self._default_log.context = context
        if disable_dmesg_log_monitor:
            self._dmesg_log = DummyLogMonitor()
        else:
            self._dmesg_log = DmesgLogMonitor()

        self.ovms_ports = {}
        self.remote_ip = remote_ip
        if grpc_port:
            self.ovms_ports[GRPC] = grpc_port
        if rest_port:
            self.ovms_ports[REST] = rest_port
        self.target_device = target_device
        self.target_device_lock_file = lock_file
        self.full_command = full_command

    def get_break_msg_list(self, models):
        break_msg_list = [
            CPP_STD_EXCEPTION,
            OvmsMessages.ERROR_CANNOT_COMPILE_MODEL_INTO_TARGET_DEVICE,
            OvmsMessages.CANNOT_OPEN_LIBRARY,
            OvmsMessages.ERROR_FAILED_TO_PARSE_SHAPE_SHORT,
            OvmsMessages.ERROR_LOADING_PRECONDITION_FAILED,
            OvmsMessages.ERROR_DURING_LOADING_INPUT_TENSORS,
        ]

        custom_node_present = any(map(lambda x: getattr(x, "child_nodes", None), models))
        if custom_node_present:
            break_msg_list.extend([
                OvmsMessages.PIPELINE_REFERS_TO_INCORRECT_LIBRARY,
            ])

        if any([x.is_mediapipe for x in models]):
            break_msg_list.extend([
                OvmsMessages.MEDIAPIPE_FAILED_TO_OPEN_GRAPH_SHORT,
            ])

        custom_loader_present = any(map(lambda x: x.custom_loader is not None, models))
        if custom_loader_present:
            break_msg_list.extend([
                OvmsMessages.CUSTOM_LOADER_INVALID_CUSTOM_LOADER_OPTIONS,
            ])

        if any(map(lambda x: x.is_on_cloud, models)):
            break_msg_list.extend([
                OvmsMessages.ERROR_FAILED_TO_CONNECT_TO_ANY_PROXY_ENDPOINT,
                OvmsMessages.S3_WRONG_AUTHORIZATION,
                OvmsMessages.GS_WRONG_AUTHORIZATION,
            ])

        if ct.is_plugin_target():
            break_msg_list.extend(
                [OvmsMessages.ERROR_FAILED_TO_CREATE_PLUGIN, OvmsMessages.ERROR_FAILED_TO_LOAD_LIBRARY]
            )

        if all([x.is_hf_direct_load and not x.is_local for x in models]):
            break_msg_list.extend([
                OvmsMessages.WARNING_NO_VERSION_FOUND_FOR_MODEL,
            ])

        return break_msg_list

    def ensure_models_loaded(self, models: List[ModelInfo] = None, timeout=None):
        callbacks = []
        if machine_is_reserved_for_test_session and is_single_threaded():
            callbacks += [self._dmesg_log.raise_on_unexpected_messages]
        custom_msg_list = []

        # Prepare messages for mapping.
        for model in models:
            if model.use_mapping is True:
                mapping_dict = model.get_mapping_dict(self.container_folder)
                msg = self._default_log.get_log_models_mapping_messages(model, mapping_dict, shapeless=True)
                custom_msg_list.extend(msg)

        self._default_log.models_loaded(
            models,
            break_msg_list=self.get_break_msg_list(models),
            custom_msg_list=custom_msg_list,
            timeout=timeout,
            callbacks=callbacks,
            ovms_instance=self,
        )

    def ensure_models_unloaded(self, models: List[ModelInfo] = None, timeout=None):
        self._default_log.models_unloaded(models, timeout=timeout)

    def ensure_started(
        self,
        expected_loaded_models: List[ModelInfo] = None,
        custom_messages: list = None,
        expected_unloaded_models: List[ModelInfo] = None,
        timeout: int = None,
        os_type: str = None,
    ):
        self.wait_for_status(CONTAINER_STATUS_RUNNING, break_status=CONTAINER_STATUS_EXITED)

        if expected_loaded_models is not None:
            expected_models = []
            for model in expected_loaded_models:
                if isinstance(model, Pipeline):
                    if not model.is_mediapipe:
                        # To avoid double model addition (get_models() returns [self] for MediaPipe)
                        expected_models.extend(model.get_regular_models())
                    expected_models.append(model)
                else:
                    expected_models.append(model)
            self.ensure_models_loaded(expected_models, timeout=timeout)

        self._default_log.models_unloaded(expected_unloaded_models, timeout=timeout, ovms_instance=self)
        self._default_log.ensure_contains_messages(custom_messages, timeout=timeout, ovms_instance=self)
        self._default_log.raise_on_unexpected_messages()
        self._default_log.reset_to_ovms_creation()
        self._default_log.is_running(self.ovms_ports, timeout, os_type)
        self._default_log.reset_to_ovms_creation()
        self._default_log.flush()

        self._dmesg_log.raise_on_unexpected_messages()

    def ensure_reloading(self, models: List[ModelInfo], timeout=None):
        self._default_log.reloading(models, timeout)

    def prepare_mediapipe_config_and_graph(self, name, params, models):
        config_dict = OvmsConfig.build(models=models)
        params.custom_config = config_dict
        params.models = models
        config_path_on_host = os.path.join(self.container_folder, Paths.MODELS_PATH_NAME)
        MediaPipeCalculator.prepare_proto_calculator(params, config_path_on_host)
        OvmsConfig.save(name, config_dict)
        return config_dict

    def update_model_list_and_config(
        self,
        name,
        models,
        models_to_verify=None,
        resources_paths=None,
        context=None,
        params=None,
        **kwargs
    ):
        self.prepare_resources(models)

        if models_to_verify:
            ovms_log = self.create_log(False)

        if models_to_verify is not None and any(model.is_mediapipe for model in models_to_verify):
            assert params is not None, "Params should be provided to create MediaPipe calculators"
            config_dict = self.prepare_mediapipe_config_and_graph(name, params, models)
        else:
            _, config_dict = OvmsConfig.generate(name, models)

        if models_to_verify:
            break_msg_list = self.get_break_msg_list(models_to_verify)
            timeout = kwargs.get("timeout", wait_for_messages_timeout)
            ovms_log.models_loaded(models_to_verify, break_msg_list=break_msg_list, ovms_instance=self, timeout=timeout)

        return config_dict

    def unload_all_models(self):
        self.update_model_list_and_config(self.name, [])

    def get_port(self, api_type):
        if isinstance(api_type, str):
            return self.ovms_ports[api_type]
        else:
            return self.ovms_ports[api_type.type]

    def execute_and_check(self, cmd, verbose=False, cwd=None):
        exit_code, stdout = self.execute_command(cmd, cwd)
        assert exit_code == 0, f"Unexpected return code: {exit_code} during executing cmd: {cmd}\n\tOutput: {stdout}"
        if verbose:
            logger.info(stdout)
        return stdout

    def get_env_variables(self):
        return self.container.kwargs.get("environment", [])

    def install_package(self, base_os, tool_name):
        environment = self.get_env_variables()
        tool_installed = False

        for i in range(3):
            try:
                if i != 0:
                    logger.info(f"Running retry {i}")
                    sleep(10)

                cmd = PackageManager.create(base_os).get_install_cmd(tool_name)
                if environment is None or "http_proxy" not in environment:
                    self.execute_and_check(
                        f"/bin/bash -c "
                        f"'export http_proxy={container_proxy} https_proxy={container_proxy}"
                        f" && {cmd}'"
                    )
                else:
                    self.execute_and_check(f"/bin/bash -c '{cmd}'")

                tool_installed = True
                break
            except Exception as e:
                logger.info(f"Exception received. {e}")
                sleep(5)

        assert tool_installed, f"Failed to install {tool_name} in container"

    def get_logs(self):
        return self._default_log.get_all_logs()

    def get_logs_as_txt(self):
        return self._default_log.get_logs_as_txt()

    def change_log_monitor(self, log_monitor):
        self._default_log = log_monitor

    def wait_for_container_status_exited(self):
        try:
            self.wait_for_status(status=CONTAINER_STATUS_EXITED)
        except Exception as e:
            logger.error("Container cannot close properly")
            logger.exception(str(e))
            raise DockerCannotCloseProperly(str(e))

    def wait_for_status(self, status: str = CONTAINER_STATUS_RUNNING, break_status: str = None, timeout=60):
        ovms_logs_lines = None
        end_time = datetime.now() + timedelta(seconds=timeout)
        while datetime.now() < end_time:
            current_status = self.get_status(status)
            if break_status and current_status == break_status:
                ovms_logs_lines = self.get_logs()
                result = get_exception_by_ovms_log(ovms_logs_lines)
                if result:
                    exception, line = result
                    raise exception(line)
                else:
                    raise OvmsTestException(f"Received break status: {current_status}", ovms_log=ovms_logs_lines)
            if current_status == status:
                break

    def ensure_status(self, status: str = CONTAINER_STATUS_RUNNING):
        assert self.container.ensure_status(status)

    @abstractmethod
    def execute_command(self, cmd, stream=False, cwd=None):
        pass

    def prepare_resources(self, models):
        """
        Execute on each model method prepare_resources (copy models, prepare custom_nodes and custom libraries if required)

        Parameters:
        models (List[ModelInfo]): list of objects that needs to be prepared (ModelInfo, Pipeline, CustomNode, CustomLibrary)
        """
        TestEnvironment.current.prepare_container_folders(self.name, models)

    def create_log(self, reset_to_ovms_creation, wait_for_log_timeout=None, use_default_logger=False) -> LogMonitor:
        log = self._default_log if use_default_logger else self._create_logger()
        log.get_all_logs()
        if wait_for_log_timeout is not None:
            start = datetime.now()
            while not log._read_lines and (datetime.now() - start).total_seconds() <= wait_for_log_timeout:
                log.get_all_logs()
                sleep(1)
        if not reset_to_ovms_creation:
            log.flush()
        log.logger_creation_start_offset = log.current_offset
        if not log._read_lines:
            log._read_lines = self._default_log._read_lines[:]    # fix for CVS-126060
        return log

    def get_dmesg_log_monitor(self):
        return self._dmesg_log

    def fetch_and_store_ovms_pid(self, timeout=60):
        """
        Fetch and save OVMS process id.
        Expect that child class object will provide `self.ovms_pid`
        """
        self._dmesg_log.ovms_pid = self.ovms_pid

    @abstractmethod
    def fetch_and_store_ovms_pid(self, timeout=10):
        """
        Fetch and save OVMS process id.
        """
        pass

    @abstractmethod
    def start(self, ensure_started=False, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _create_logger(self) -> LogMonitor:
        raise NotImplementedError()

    @staticmethod
    def get_signal_type(terminate_signal_type):
        if terminate_signal_type == Ovms.SIGKILL_SIGNAL:
            return signal.SIGKILL
        elif terminate_signal_type == Ovms.SIGINT_SIGNAL:
            return signal.SIGINT
        elif terminate_signal_type == Ovms.SIGTERM_SIGNAL:
            return signal.SIGTERM
        else:
            raise NotImplementedError(f"Unknown signal: {terminate_signal_type}")

    def filter_unexpected_messages(self, unexpected_messages):
        for msg in unexpected_messages:
            # 1) This message is expected in OV tests with batch_size=0
            # 2) This message can happen in Cython instances that use threading # CVS-165321
            if all([
                OvmsMessages.ERROR_TERMINATE_CALLED in msg,
                self.context is not None and "test_ov_app_cpp_batch_size_0" in self.context.name,
            ]) or all([
                OvmsMessages.ERROR_TERMINATE_CALLED in msg,
                OvmsMessages.STD_SYSTEM_ERROR in msg,
                hasattr(self, "cmd") and self.cmd is not None and "OvmsCapiInstance" in self.cmd.__str__,
            ]):
                unexpected_messages.remove(msg)
        return unexpected_messages

    def cleanup(self, timeout=30):
        try:
            if self.target_device_lock_file and self.target_device_lock_file.is_locked:
                self.target_device_lock_file.release()
        except Exception as e:
            logger.exception(e)

        if artifacts_dir != "":
            unexpected_messages = []
            for log_type, log in [("ovms", self._default_log), ("dmesg", self._dmesg_log)]:
                log_name = f"{log_type}_{self.name}.log"
                _, log_messages = log.cleanup(log_name)
                if log_messages is not None:
                    unexpected_messages.extend(log_messages)
                unexpected_messages = self.filter_unexpected_messages(unexpected_messages)
                if unexpected_messages:
                    logger.error(f"Errors found in {log_type}: {unexpected_messages}")
                    dmesg_exceptions = get_children_from_module(DmesgError, assertions_module)
                    error_message = (
                        f"Found unexpected messages in log files after OVMS instance cleanup: {unexpected_messages}"
                    )
                    ovms_log = self._default_log.get_all_logs()
                    dmesg_log = self._dmesg_log.get_all_logs()
                    for name, exception_class in dmesg_exceptions:
                        msg = getattr(exception_class, "msg", None)
                        for m in unexpected_messages:
                            if m in msg:
                                raise exception_class(error_message, ovms_log, dmesg_log)
                    raise UnwantedMessageError(error_message, ovms_log, dmesg_log)

        if self.container_folder is not None and os.path.exists(self.container_folder):
            error = None
            start = datetime.now()
            while (datetime.now() - start).total_seconds() <= timeout:
                try:
                    shutil.rmtree(self.container_folder)
                except PermissionError as e:
                    error = e
                    sleep(1)
                else:
                    break
            else:
                if (hasattr(self, "cmd") and self.cmd is not None and self.cmd.base_os == OsType.Windows
                        and type(error) == PermissionError):
                    # workaround for Windows: https://jira.devtools.intel.com/browse/CVS-161953
                    change_dir_permissions(self.container_folder)
                    shutil.rmtree(self.container_folder)
                else:
                    raise error

    def release_ports(self):
        for api in [REST, GRPC]:
            port = self.ovms_ports.pop(api, None)
            if port:
                logger.info(f"Releasing {api} port")
                PortManager(api).release_port(port)

    @staticmethod
    def acquire_target_device_lock(target_device):
        target_device = target_device.strip("'").split(" ")[0] if type(target_device) == str else target_device
        max_locks = MAX_WORKERS_PER_TARGET_DEVICE[target_device]
        if max_locks == 0:  # No lock required
            return None
        if max_locks == 1:
            timeout = None  # just wait patiently for your device
        else:
            timeout = 1.0  # do not wait too long, try another lock
        lock_files = [
            SelfDeletingFileLock(Paths.get_target_device_lock_file(target_device, i)) for i in range(max_locks)
        ]
        while True:
            idx = random.randint(0, max_locks - 1)
            acquired = lock_files[idx].acquire_no_raise(timeout)
            if acquired:
                return lock_files[idx]

    def stop_ovms_inside_kill(self, context, terminate_signal_type=Ovms.TERM_SIGNAL, ensure_ovms_killed=True):
        exit_code, _ = self.execute_command("ps --version")
        if exit_code != 0:
            self.install_package(base_os=context.base_os, tool_name="procps-ng")

        output = self.execute_and_check("ps aux")
        msg_list = [x for x in output.splitlines() if "strace" not in x and "xargs" not in x and "ovms/bin" in x]
        assert len(msg_list) == 1, f"Unexpected number of OVMS processes: {msg_list}"

        ovms_pid = re.search(r"^\w+\s+(\d+)", msg_list[0]).group(1)
        self.execute_and_check(f"/bin/bash -c 'kill {ovms_pid} -s {terminate_signal_type}'")

        if ensure_ovms_killed:
            try:
                cmd = f"ps -p {ovms_pid}"
                exit_code, _ = self.execute_command(cmd)
                i = 0
                ovms_log_monitor = self.create_log(True)
                while exit_code == 0:
                    logger.warning(
                        f"OVMS is still running. Tail of OVMS output: {ovms_log_monitor.get_all_logs()[:-2]}"
                    )
                    assert i < 60, f"Unable to stop OVMS process (PID: {ovms_pid})"
                    i += 1
                    sleep(1)
                    exit_code, _ = self.execute_command(cmd)
            except APIError as e:
                logger.warning(e)
                proc = Process()
                short_id = self.get_short_id()
                code, stdout, stderr = proc.run_and_check_return_all(f"docker ps -a --filter id={short_id}")
                assert short_id in stdout, f"OVMS is still running. Docker id: {short_id}, OVMS id: {ovms_pid}."


@dataclass
class OvmsRunContext:
    ovms: OvmsInstance = None
    models: List[ModelInfo] = None
    context: Context = None

    def attach_context(self, context):
        self.context = context
        context.ovms_sessions.append(self)

    def attach_resource_monitor(self, context, start=True):
        if hasattr(self.ovms, "container"):
            self.resource_monitor = DockerResourceMonitor(self.ovms.container)
            if start:
                self.resource_monitor.start()
            context.test_objects.append(self.resource_monitor)
            return self.resource_monitor
