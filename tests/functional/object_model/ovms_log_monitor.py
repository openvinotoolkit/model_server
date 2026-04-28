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

import datetime
import re
import time

import requests
from docker import from_env

from tests.functional.utils.assertions import OvmsCrashed, PodNotReadyException
from tests.functional.utils.logger import get_logger
from tests.functional.utils.process import PID_STATE_SLEEPING, PID_STATE_ZOMBIE, Process, get_pid_status
from ovms.command_wrappers.kubectl_helpers import KubeCtl
from ovms.config import is_nginx_mtls, wait_for_messages_timeout
from tests.functional.constants.core import CONTAINER_STATUS_DEAD, CONTAINER_STATUS_EXITED
from ovms.constants.custom_loader import CustomLoaderConsts
from tests.functional.constants.ovms_messages import OvmsMessages, OvmsMessagesRegex
from tests.functional.utils.log_monitor import LogMonitor

logger = get_logger(__name__)


class OvmsLogMonitor(LogMonitor):

    def _get_unexpected_messages_regex(self):
        return []

    def _get_unexpected_messages(self):
        return ["terminate called", "Exception caught in REST request handler"]

    def ensure_contains_messages(
        self,
        str_set_to_find,
        break_msg_list=None,
        timeout=None,
        callbacks=[],
        ovms_instance=None,
        all_messages=True,
    ):
        self.wait_for_messages(
            str_set_to_find,
            break_msg_list,
            raise_exception_if_not_found=True,
            timeout=timeout,
            callbacks=callbacks,
            ovms_instance=ovms_instance,
            all_messages=all_messages,
        )

    def reset_to_ovms_creation(self):
        self.current_offset = 0

    def reset_to_logger_creation(self):
        self.current_offset = self.logger_creation_start_offset

    @staticmethod
    def _calculate_batch_size_str(model):
        batch_size = model.batch_size
        if model.input_shape_for_ovms is not None:
            input_shape = model.input_shape_for_ovms
            if isinstance(input_shape, dict):
                input_shape = [x for x in input_shape.values()][0]

            if isinstance(input_shape, str):
                match = re.findall(r"([-\d:]+)", input_shape)
                if match:
                    batch_size = match[0]
                elif input_shape == "auto":
                    batch_size = 1
            elif isinstance(input_shape, list) or isinstance(input_shape, tuple):
                batch_size = int(input_shape[0])
        elif batch_size is not None and isinstance(batch_size, str) and ":" in batch_size:
            batch_size = (
                f"[{'~'.join(batch_size.split(':'))}]"  # if batch_size = "1:3" -> expected entry in OVMS log: "[1~3]"
            )
        elif model.input_shape_for_ovms is not None and isinstance(model.input_shape_for_ovms, str):
            match = re.findall(r"([-\d:]+)", model.input_shape_for_ovms)
            if match:
                batch_size = match[0]
            elif model.input_shape_for_ovms == "auto":
                batch_size = 1
        elif batch_size == "auto":
            batch_size = 1
        else:
            batch_size = model.get_expected_batch_size()
        return batch_size

    def _get_log_models_started_messages(self, models):
        result = []
        for model in models or []:
            batch_size_str = self._calculate_batch_size_str(model)
            result.append(OvmsMessages.OVMS_SERVER_RUNNING_MSG.format(model.version, model.name))
            result.append(OvmsMessages.OVMS_MODEL_LOADED.format(model.name, model.version, batch_size_str))
        return result + self._get_log_custom_loader_loaded_messages(models)

    def _get_log_models_loaded(self, models):
        result = []
        for model in models or []:
            if model.is_hf_direct_load:
                result.append(OvmsMessages.MEDIAPIPE_PIPELINE_VALIDATION_PASS_MSG.format(model.name))
            elif model.is_mediapipe:
                result.append(OvmsMessages.MEDIAPIPE_PIPELINE_VALIDATION_PASS_MSG.format(model.name))
                for reg_model in model.regular_models:
                    batch_size_str = self._calculate_batch_size_str(reg_model)
                    result.append(
                        OvmsMessages.OVMS_MODEL_LOADED.format(reg_model.name, reg_model.version, batch_size_str)
                    )
            elif model.is_pipeline():
                result.append(OvmsMessages.PIPELINE_STARTED.format(model.name))
            else:
                batch_size_str = self._calculate_batch_size_str(model)
                result.append(OvmsMessages.OVMS_MODEL_LOADED.format(model.name, model.version, batch_size_str))
        return result

    @staticmethod
    def _get_log_models_failed_to_load(models):
        phrases = [
            OvmsMessages.ERROR_EXCEPTION_CATCH,
            OvmsMessages.ERROR_UNABLE_TO_ACCESS_PATH,
            OvmsMessages.OVMS_ERROR_OCCURED_WHILE_LOADING_MODEL_GENERIC,
            OvmsMessages.OVMS_ERROR_INCORRECT_WEIGHTS_IN_BIN_FILE,
        ]
        for model in models or []:
            phrases.append(OvmsMessages.OVMS_MODEL_FAILED_TO_LOAD.format(model.name, model.version))
        return phrases

    @staticmethod
    def _get_log_models_reloading_messages(models):
        def _calculate_shape_str(shape):
            result = shape
            if ":" in in_shape:
                match_range = re.findall(r"\d+:\d+", in_shape)
                for range in match_range:
                    new_value = f"[{range.replace(':', '~')}]"
                    result = in_shape.replace(range, new_value)
            return result

        result = []
        for model in models or []:
            result.append(OvmsMessages.MODEL_RELOADING.format(model.name))
            if model.input_shape_for_ovms is not None:
                for in_name, in_shape in model.input_shape_for_ovms.items():
                    log_shape_str = _calculate_shape_str(in_shape)
                    result.append(OvmsMessages.MODEL_INPUT_SHAPE_RELOADING.format(in_name, in_name, log_shape_str))
        return result

    @staticmethod
    def get_log_models_mapping_messages(model, mapping_dict, reload=False, shapeless=False):
        def _calculate_shape_str(shape):
            return str(tuple(shape["shape"])).replace(" ", "")  # [1, 1, 1, 1] -> (1,1,1,1)

        result = []
        if reload:
            result.append(OvmsMessages.MODEL_RELOADING.format(model.name))
        for (_, in_shape), (in_key, in_value) in zip(model.inputs.items(), mapping_dict["inputs"].items()):
            if shapeless:
                result.append(OvmsMessages.MODEL_INPUT_NAME_MAPPING_NAME.format(in_key, in_value))
            else:
                log_input_shape_str = _calculate_shape_str(in_shape)
                result.append(OvmsMessages.MODEL_INPUT_SHAPE_RELOADING.format(in_key, in_value, log_input_shape_str))
        for (_, out_shape), (out_key, out_value) in zip(model.outputs.items(), mapping_dict["outputs"].items()):
            if shapeless:
                result.append(OvmsMessages.MODEL_OUTPUT_NAME_MAPPING_NAME.format(out_key, out_value))
            else:
                log_output_shape_str = _calculate_shape_str(out_shape)
                result.append(
                    OvmsMessages.MODEL_OUTPUT_SHAPE_RELOADING.format(out_key, out_value, log_output_shape_str)
                )

        return result

    @staticmethod
    def _get_log_models_unloaded_messages(models):
        result = []
        for model in models or []:
            if model.is_mediapipe:
                result.append(OvmsMessages.MEDIAPIPE_UNLOADED.format(model.name))
                for reg_model in model.regular_models:
                    result.append(OvmsMessages.OVMS_SERVER_UNLOADED_MSG.format(reg_model.version, reg_model.name))
            else:
                result.append(OvmsMessages.OVMS_SERVER_UNLOADED_MSG.format(model.version, model.name))
        return result

    @staticmethod
    def _get_log_pipelines_started_messages(pipelines):
        result = []
        for pipeline in pipelines:
            result.append(OvmsMessages.PIPELINE_STARTED.format(pipeline.name))
        return result

    @staticmethod
    def _get_log_pipelines_unloaded_messages(pipelines):
        result = []
        for pipeline in pipelines:
            if pipeline.is_mediapipe:
                result.append(OvmsMessages.MEDIAPIPE_UNLOADED.format(pipeline.name))
            else:
                result.append(OvmsMessages.PIPELINE_UNLOADED.format(pipeline.name))
        return result

    @staticmethod
    def _get_log_custom_loader_loaded_messages(models):
        result = []
        custom_loaders = set()  # using set instead list will help avoid redundancy

        for model in models or []:
            if model.custom_loader is not None:
                custom_loaders.add(model.custom_loader)

        for cs in custom_loaders:
            loader_loaded_logs = CustomLoaderConsts.get_logs_for_loaded_loader(cs.name)
            result.extend(loader_loaded_logs)
        return result

    def started(self, models, pipelines=None, timeout=None):
        msg_list = self._get_log_models_started_messages(models)
        break_msg_list = self._get_log_models_failed_to_load(models)
        if pipelines is not None:
            msg_list.extend(self._get_log_pipelines_started_messages(pipelines))
        if timeout is None:
            timeout = 60
            if models:
                timeout += sum([model.get_ovms_loading_time() for model in models])
        self.ensure_contains_messages(msg_list, break_msg_list, timeout=timeout)

    def reloading(self, models, timeout=30):
        msg_list = self._get_log_models_reloading_messages(models)
        self.ensure_contains_messages(msg_list, timeout=timeout)

    def model_mapping_loaded(self, model, mapping_dict, timeout=30, reload=False):
        msg_list = self.get_log_models_mapping_messages(model, mapping_dict, reload=reload)
        start_time = time.time()
        # model mapping is loading: should be fast since txt data is loaded.
        self.ensure_contains_messages(msg_list, timeout=timeout)
        timeout -= time.time() - start_time

        # model binaries is loading: could be time-consuming.
        self.models_loaded([model], timeout=timeout)

    def models_unloaded(self, models, pipelines=None, timeout=None, ovms_instance=None):
        msg_list = self._get_log_models_unloaded_messages(models)
        if pipelines is not None:
            msg_list.extend(self._get_log_pipelines_unloaded_messages(pipelines))
        if timeout is None:
            timeout = 60
            if models:
                timeout += sum([model.get_ovms_loading_time() for model in models])
        self.ensure_contains_messages(msg_list, timeout=timeout, ovms_instance=ovms_instance)

    def models_loaded(
        self, models, custom_msg_list=None, break_msg_list=None, timeout=None, callbacks=[], ovms_instance=None
    ):
        if timeout is None:
            timeout = wait_for_messages_timeout
            if models:
                timeout += sum([model.get_ovms_loading_time() for model in models])
        msg_list = self._get_log_models_loaded(models)
        if custom_msg_list is not None:
            msg_list.extend(custom_msg_list)
        self.ensure_contains_messages(
            msg_list, break_msg_list, timeout=timeout, callbacks=callbacks, ovms_instance=ovms_instance
        )

    def custom_loader_loaded(self, models, timeout=30):
        msg_list = self._get_log_custom_loader_loaded_messages(models)
        self.ensure_contains_messages(msg_list, timeout=timeout)

    def action_on_models(
        self, models_loaded=None, models_unloaded=None, models_reloaded=None, custom_msg_list=None, timeout=30
    ):
        msg_list = self._get_log_models_started_messages(models_loaded)
        msg_list += self._get_log_models_unloaded_messages(models_unloaded)
        msg_list += self._get_log_models_reloading_messages(models_reloaded)
        if custom_msg_list:
            msg_list += custom_msg_list
        self.ensure_contains_messages(msg_list, timeout=timeout)

    def is_running(self, ovms_ports, timeout, os_type=None):
        msg_list = [OvmsMessages.OVMS_SERVICES_RUNNING_MSG[key] for key in ovms_ports]
        if is_nginx_mtls:
            msg_list += [OvmsMessages.NGINX_STARTED_WITH_PID]
        break_msg_list = [OvmsMessages.OVMS_STOPPING]
        self.ensure_contains_messages(msg_list, break_msg_list, timeout=timeout)

    @staticmethod
    def get_message_time(msg):
        # message example '[2022-01-20 16:45:47.495][1][serving][info][modelinstance.cpp:718] Loaded model'
        msg_time_str = msg.split("[")[1].strip("]")
        msg_time = datetime.datetime.strptime(msg_time_str, "%Y-%m-%d %H:%M:%S.%f")
        return msg_time

    def get_models_loading_time(self, models, is_reload):
        result = []
        for model in models:
            if is_reload:
                found_messages, messages_to_find_vs_results_map = self.find_messages(
                    [OvmsMessages.MODEL_RELOADING.format(model.name)], raise_exception_if_not_found=True
                )
            else:
                found_messages, messages_to_find_vs_results_map = self.find_messages(
                    [OvmsMessages.MODEL_LOADING.format(model.name, model.version, model.base_path)],
                    raise_exception_if_not_found=True,
                )
            model_loading_msg = list(messages_to_find_vs_results_map.values())[0]
            start_loading_time = self.get_message_time(model_loading_msg)

            start_msg_list = self._get_log_models_loaded([model])
            found_messages, messages_to_find_vs_results_map = self.find_messages(
                start_msg_list, raise_exception_if_not_found=True
            )
            model_loaded_msg = list(messages_to_find_vs_results_map.values())[0]
            finished_loading_time = self.get_message_time(model_loaded_msg)
            loading_time = finished_loading_time - start_loading_time
            result.append((model, loading_time))
            logger.info(f"{model} model loading time: {loading_time}")

        return result

    @staticmethod
    def find_no_of_infer_requests(log_lines):
        string_to_find = "] Loaded model"
        for i, ll in enumerate(log_lines):
            if string_to_find in ll:
                noir = int(log_lines[i].split()[-1])
        return noir

    @staticmethod
    def get_status_change_from_logs(logs, expected_state=""):
        status_change_messages = list(filter(lambda x: OvmsMessagesRegex.STATUS_CHANGE_RE.search(x), logs))
        status_change_messages = list(filter(lambda x: expected_state in x, status_change_messages))
        return status_change_messages

    def get_log_value(self, msg_to_found):
        all_messages_found, messages_to_find_vs_results_map = self.find_messages(
            [msg_to_found], raise_exception_if_not_found=True
        )
        assert all_messages_found
        # Searching for log phrase, e.g.:
        # "Number of OpenVINO streams: 11" or "Number of OpenVINO streams: 4"
        # "No of InferRequests: 11" "No of InferRequests: 4"
        # from the whole log line, e.g.:
        # "["2022-12-08 12:09:17.212][1][serving][info][modelinstance.cpp:790] Loaded model resnet-50-tf; version: 1; batch size: 1; No of InferRequests: 4"
        phrase = re.search(rf"{msg_to_found} \d+", messages_to_find_vs_results_map[msg_to_found]).group()
        value = re.search(r"\d+", phrase).group()  # searching for single value
        return value


class BinaryOvmsLogMonitor(OvmsLogMonitor):
    def __init__(self, ovms_process, **kwargs):
        super().__init__(**kwargs)
        self._proc = ovms_process

    def is_ovms_running(self):
        status = get_pid_status(self._proc._proc.pid)
        if status in [PID_STATE_ZOMBIE]:
            return False
        if status in [PID_STATE_SLEEPING]:
            return True
        return True

    def get_all_logs(self):
        stdout, stderr = self._proc.get_output()
        if stderr:
            logger.error(
                f"Detect non-empty stderr! It is recommended to redirect stderr to stdout: 2>&1. STDERR: {stderr}"
            )
        self._read_lines += stdout.splitlines()
        return self._read_lines


class OvmsDockerLogMonitor(OvmsLogMonitor):
    @staticmethod
    def create(container_id):
        client = from_env()
        container = client.containers.get(container_id)
        return OvmsDockerLogMonitor(container)

    def __init__(self, container, **kwargs):
        super().__init__(**kwargs)
        self._container = container

    def get_all_logs(self):
        try:
            self._read_lines = self._container.logs().decode().splitlines()
        except requests.exceptions.HTTPError as e:
            raise OvmsCrashed(msg=str(e), ovms_log=self.get_logs_as_txt())
        return self._read_lines

    def get_logs_as_txt(self):
        process = Process()
        process.disable_check_stderr()
        exit_code, stdout, _ = process.run(f"docker logs {self._container.id} 2>&1")
        return stdout

    def is_ovms_running(self):
        self._container.reload()
        return self._container.status not in [CONTAINER_STATUS_EXITED, CONTAINER_STATUS_DEAD]


class OvmsDockerStreamLogMonitor(OvmsDockerLogMonitor):
    def __init__(self, container, output_steam, **kwargs):
        super().__init__(container, **kwargs)
        self._output_steam = output_steam

    def get_all_logs(self):
        if self._container.container.get_status() != CONTAINER_STATUS_EXITED:
            logger.warning("Attempt for getting logs from unclosed docker, reading stream could lead to deadlock")
            return []
        self._read_lines = list(map(lambda x: x.decode().strip(), self._output_steam))
        return self._read_lines


class OvmsCmdLineDockerLogMonitor(OvmsLogMonitor):

    def __init__(self, docker_id, **kwargs):
        super().__init__(**kwargs)
        self.docker_id = docker_id
        self.process = Process()
        self.process.set_log_silence()

    def get_all_logs(self):
        exit_code, stdout, _ = self.process.run(f"docker logs {self.docker_id} 2>&1")
        self._read_lines = stdout.splitlines()
        return self._read_lines


class OvmsPodLogMonitor(OvmsLogMonitor):
    def __init__(self, kube_client, pod_name, namespace, **kwargs):
        super().__init__(**kwargs)
        self.kube_client = kube_client
        self.pod_name = pod_name
        self.namespace = namespace

    def get_all_logs(self):
        raw_logs = KubeCtl.get_logs(
            self.kube_client,
            pod_whole_name=self.pod_name,
            namespace=self.namespace,
            print_logs=False,
            ignore_exception=True,
        )
        if raw_logs:
            self._read_lines = raw_logs.splitlines()
        return self._read_lines

    def is_ovms_running(self):
        try:
            KubeCtl.ensure_pod_is_ready(self.kube_client, self.pod_name, self.namespace)
            return True
        except (TimeoutError, PodNotReadyException) as e:
            raise Exception(f"OVMS '{self.pod_name}' is not ready: {e}")
