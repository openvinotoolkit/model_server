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

from dataclasses import dataclass
from typing import Union

from tests.functional.utils.logger import get_logger
from tests.functional.constants.os_type import OsType
from tests.functional.config import enable_plugin_config_target_device
from tests.functional.constants.metrics import MetricsPolicy
from tests.functional.constants.ovms import Ovms, set_plugin_config_boolean_value
from tests.functional.constants.ovms_openai import ImagesRequestParamsValues
from tests.functional.constants.paths import Paths

logger = get_logger(__name__)


def create_ovms_command(
        config_path,
        model_path,
        model_name,
        parameters,
        cpu_extension_path,
        batch_size=None,
        shape=None,
        ovms_type=None,
        base_os=None,
        pull=None,
        source_model=None,
        gguf_filename=None,
        model_repository_path=None,
        task=None,
        task_params=None,
        list_models=None,
        overwrite_models=None,
        add_to_config=False,
        remove_from_config=False,
        single_mediapipe_model_mode=False,
        resolution=None,
        cache_size=None,
        pooling=None,
):
    is_stateful = parameters.is_stateful_model_present() if parameters.is_stateful is None else parameters.is_stateful
    layout = parameters.get_layout_from_regular_models()

    common_parameters = {
        "grpc_port": parameters.grpc_port,
        "rest_port": parameters.rest_port,
        "logging_level": parameters.log_level,
        "layout": layout,
        "stateful": is_stateful,
        "rest_workers": parameters.rest_workers,
        "grpc_workers": parameters.grpc_workers,
        "check_version": parameters.check_version,
        "file_system_poll_wait_seconds": parameters.file_system_poll_wait_seconds,
        "cpu_extension": cpu_extension_path,
        "metrics_enable": parameters.metrics_enable,
        "metrics_list": parameters.metrics_list,
        "sequence_cleaner_poll_wait_minutes": parameters.sequence_cleaner_poll_wait_minutes,
        "ovms_type": ovms_type,
        "base_os": base_os,
        "allowed_local_media_path": parameters.allowed_local_media_path,
        "allowed_media_domains": parameters.allowed_media_domains,
        "pooling": pooling,
    }
    pull_parameters = {
        "pull": pull,
        "source_model": source_model,
        "gguf_filename": gguf_filename,
        "model_repository_path": model_repository_path,
        "task": task,
        "task_params": task_params,
        "list_models": list_models,
        "overwrite_models": overwrite_models,
        "add_to_config": add_to_config,
        "remove_from_config": remove_from_config,
        "resolution": resolution,
        "cache_size": cache_size,
    }
    if config_path is not None:
        if add_to_config or remove_from_config:
            return OvmsCommand(config_path=config_path, model_name=model_name, **common_parameters, **pull_parameters)
        return OvmsCommand(config_path=config_path, **common_parameters)
    else:
        plugin_config = parameters.get_plugin_config_from_regular_models()
        if enable_plugin_config_target_device:
            plugin_config_target_device = Ovms.PLUGIN_CONFIG[parameters.target_device]
            plugin_config = (
                {**plugin_config, **plugin_config_target_device}
                if plugin_config is not None
                else {**plugin_config_target_device}
            )
        use_parameter = not any([single_mediapipe_model_mode, list_models, add_to_config, remove_from_config])
        return OvmsCommand(
            model_path=model_path,
            model_name=model_name,
            plugin_config=plugin_config if use_parameter else None,
            batchsize=batch_size if use_parameter else None,
            nireq=parameters.nireq if use_parameter else None,
            target_device=parameters.target_device if use_parameter else None,
            shape=shape if use_parameter else None,
            model_version_policy=parameters.model_version_policy if use_parameter else None,
            max_sequence_number=parameters.max_sequence_number if use_parameter else None,
            idle_sequence_cleanup=parameters.idle_sequence_cleanup if use_parameter else None,
            low_latency_transformation=parameters.low_latency_transformation if use_parameter else None,
            **common_parameters,
            **pull_parameters,
        )


@dataclass
class OvmsCommand(object):
    logging_level: str = None
    model_path: str = None
    model_name: str = None
    plugin_config: Union[dict, str] = None
    grpc_port: int = None
    grpc_workers: int = None
    nireq: int = None
    target_device: str = None
    batchsize: Union[int, str] = None
    config_path: str = None
    rest_port: int = None
    rest_workers: int = None
    shape: str = None
    model_version_policy: str = None
    file_system_poll_wait_seconds: str = None
    max_sequence_number: int = None
    sequence_cleaner_poll_wait_minutes: str = None
    low_latency_transformation: bool = None
    stateful: bool = False
    check_version: bool = False
    layout: str = None
    cpu_extension: str = None
    idle_sequence_cleanup: bool = None
    metrics_enable: MetricsPolicy = MetricsPolicy.NotDefined
    metrics_list: list = None
    ovms_type: str = None
    base_os: str = None
    allowed_local_media_path: str = None
    allowed_media_domains: str = None
    pull: bool = False
    source_model: str = None
    gguf_filename: str = None
    model_repository_path: str = None
    task: str = None
    task_params: str = None
    list_models: bool = None
    overwrite_models: bool = None
    add_to_config: bool = False
    remove_from_config: bool = False
    resolution: str = None
    cache_size: int = None
    pooling: str = None

    def to_list(self):
        ovms_command, _ = Ovms.get_ovms_binary_paths(self.ovms_type, self.base_os)
        command_parts = [ovms_command]

        if self.check_version:
            command_parts.append("--version")
        else:
            if self.pull:
                command_parts.append("--pull")

            if self.source_model is not None:
                command_parts.append("--source_model")
                command_parts.append(self.source_model)
                if self.gguf_filename is not None:
                    command_parts.append("--gguf_filename")
                    command_parts.append(self.gguf_filename)

            if self.model_repository_path is not None:
                command_parts.append("--model_repository_path")
                command_parts.append(self.model_repository_path)

            if self.task is not None:
                command_parts.append("--task")
                command_parts.append(self.task)

                if self.task_params is not None:
                    command_parts.append("--task_params")
                    command_parts.append(self.task_params)

            if self.list_models:
                command_parts.append("--list_models")

            if self.overwrite_models:
                command_parts.append("--overwrite_models")

            if self.add_to_config:
                command_parts.append("--add_to_config")

            if self.remove_from_config:
                command_parts.append("--remove_from_config")

            if self.resolution is not None:
                command_parts.append("--resolution")
                command_parts.append(self.resolution)

            if self.cache_size is not None:
                command_parts.append("--cache_size")
                command_parts.append(str(self.cache_size))

            if self.model_path and self.config_path:
                logger.debug("Both config_path and model_path with model_name not set!!!")

            if self.logging_level is not None:
                command_parts.append("--log_level")
                command_parts.append(str(self.logging_level))

            if self.grpc_port is not None:
                command_parts.append("--port")
                command_parts.append(str(self.grpc_port))

            if self.grpc_workers is not None:
                command_parts.append("--grpc_workers")
                command_parts.append(str(self.grpc_workers))

            if self.nireq is not None:
                command_parts.append("--nireq")
                command_parts.append(str(self.nireq))

            if self.rest_workers is not None:
                command_parts.append("--rest_workers")
                command_parts.append(str(self.rest_workers))

            if self.rest_port is not None:
                command_parts.append("--rest_port")
                command_parts.append(str(self.rest_port))

            if self.config_path is not None:
                command_parts.append("--config_path")
                command_parts.append(self.config_path)

            if self.model_path is not None:
                command_parts.append("--model_path")
                command_parts.append(self.model_path)

            if self.model_name is not None:
                command_parts.append("--model_name")
                command_parts.append(self.model_name)

            if self.target_device is not None:
                command_parts.append("--target_device")
                command_parts.append(f'"{str(self.target_device)}"' if
                                     self.target_device == ImagesRequestParamsValues.MIXED_NPU_DEVICE
                                     else str(self.target_device))

            if self.batchsize is not None:
                command_parts.append("--batch_size")
                command_parts.append(str(self.batchsize))

            if self.allowed_local_media_path is not None:
                command_parts.append("--allowed_local_media_path")
                images_path = Paths.IMAGES_PATH_INTERNAL if self.base_os != OsType.Windows \
                    else self.allowed_local_media_path
                command_parts.append(images_path)

            if self.allowed_media_domains is not None:
                command_parts.append("--allowed_media_domains")
                command_parts.append(self.allowed_media_domains)

            if self.plugin_config is not None:
                command_parts.append("--plugin_config")
                plugin_config_str = str(self.plugin_config).replace('"', '\\"').replace("'", '\\"')
                plugin_config_str = set_plugin_config_boolean_value(plugin_config_str)
                command_parts.append(f"\"{plugin_config_str}\"")

            if self.shape is not None:
                command_parts.append("--shape")
                shape_str = str(self.shape).replace('"', '\\"').replace("'", '\\"')
                if isinstance(self.shape, dict):
                    shape_str = shape_str.replace("(", '\\"(').replace(")", ')\\"')
                command_parts.append(f"\"{shape_str}\"")

            if self.model_version_policy is not None:
                command_parts.append("--model_version_policy")
                model_version_policy_str = str(self.model_version_policy).replace('"', '\\"').replace("'", '\\"')
                command_parts.append(f"\"{model_version_policy_str}\"")

            if self.file_system_poll_wait_seconds is not None:
                command_parts.append("--file_system_poll_wait_seconds")
                command_parts.append(str(self.file_system_poll_wait_seconds))

            if self.sequence_cleaner_poll_wait_minutes is not None:
                command_parts.append("--sequence_cleaner_poll_wait_minutes")
                command_parts.append(str(self.sequence_cleaner_poll_wait_minutes))

            if self.idle_sequence_cleanup is not None:
                command_parts.append("--idle_sequence_cleanup")
                command_parts.append(str(self.idle_sequence_cleanup))

            if self.max_sequence_number is not None:
                command_parts.append("--max_sequence_number")
                command_parts.append(str(self.max_sequence_number))

            if self.low_latency_transformation is not None:
                command_parts.append("--low_latency_transformation")
                command_parts.append(str(self.low_latency_transformation))

            if self.layout:
                command_parts.append("--layout")
                command_parts.append(str(self.layout))

            if self.cpu_extension:
                command_parts.append("--cpu_extension")
                command_parts.append(str(self.cpu_extension))

            if self.stateful:
                command_parts.append("--stateful")

            if self.metrics_enable == MetricsPolicy.EnabledInCli:
                command_parts.append("--metrics_enable")

                if self.metrics_list:
                    command_parts.append(f'--metrics_list {",".join(self.metrics_list)}')

            if self.metrics_enable == MetricsPolicy.EnabledMetricsList and self.metrics_list is not None:
                command_parts.append(f'--metrics_list {",".join(self.metrics_list)}')

            if self.pooling is not None:
                command_parts.append("--pooling")
                command_parts.append(str(self.pooling))

        return command_parts

    def __str__(self):
        return " ".join(self.to_list()).replace(" False", "=False").replace(" True", "=True")
