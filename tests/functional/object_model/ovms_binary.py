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

import json
import os
import psutil
from datetime import datetime
from pathlib import Path

from tests.functional.utils.context import Context
from tests.functional.utils.logger import get_logger
from tests.functional.constants.os_type import OsType
from tests.functional.utils.process import Process

from ovms.command_wrappers.ovms import create_ovms_command
from tests.functional.constants.core import CONTAINER_STATUS_EXITED, CONTAINER_STATUS_RUNNING
from tests.functional.constants.models import Muse
from tests.functional.constants.ovms_binaries import get_ovms_binary_cmd_setup
from tests.functional.constants.ovms_type import OvmsType
from tests.functional.constants.paths import Paths
from tests.functional.utils.log_monitor import LogMonitor
from tests.functional.object_model.cpu_extension import MuseModelExtension
from ovms.object_model.mediapipe_calculators import MediaPipeCalculator
from ovms.object_model.ovms_config import OvmsConfig
from tests.functional.object_model.ovms_docker import OvmsDockerLauncher, OvmsDockerParams
from tests.functional.object_model.ovms_instance import OvmsInstance, OvmsRunContext
from ovms.object_model.ovms_log_monitor import BinaryOvmsLogMonitor
from tests.functional.object_model.test_environment import TestEnvironment
from ovms.remote_test_environment import copy_custom_lib_to_host

logger = get_logger(__name__)


def start_binary_ovms(
        context: Context, parameters: OvmsDockerParams, path_to_binary_ovms, environment: dict = None, **kwargs
):
    resources_dir, _ = TestEnvironment.current.prepare_container_folders(parameters.name, parameters.get_models())

    if parameters.target_device is None:
        parameters.target_device = context.target_device

    if environment is not None:
        for key, value in environment.items():
            os.environ[key] = value

    # Automatically set use_config if the following conditions are met
    use_config = (
        bool(parameters.use_config)
        or bool(parameters.custom_config)
        or (parameters.models is not None
            and (any([model.is_pipeline() for model in parameters.models])
                 or len(parameters.models) > 1
                 or any(model.is_mediapipe and not model.single_mediapipe_model_mode for model in parameters.models)))
    )

    config_path_on_host = None
    config_dir_path_on_host = None
    if use_config:
        config_dir_path_on_host, _ = OvmsDockerLauncher.create_config(parameters, parameters.name)
        config_path_on_host = os.path.join(config_dir_path_on_host, Paths.CONFIG_FILE_NAME)
        OvmsConfig.replace_config_models_paths_for_binary(
            context,
            config_path_on_host,
            resources_dir,
            parameters.name,
            **kwargs,
        )

    if parameters.use_subconfig:
        if use_config:
            config_dict = OvmsConfig.load(config_path_on_host)
            for mediapipe_model in config_dict['mediapipe_config_list']:
                subconfig_path = os.path.join(
                    config_dir_path_on_host, mediapipe_model["name"],
                    os.path.basename(mediapipe_model["subconfig"])
                ) if "subconfig" in mediapipe_model else \
                    os.path.join(config_dir_path_on_host, mediapipe_model["name"], Paths.SUBCONFIG_FILE_NAME)
                OvmsConfig.replace_subconfig_paths(parameters.name, subconfig_path, resources_dir)
        else:
            config_dir_path_on_host = os.path.join(TestEnvironment.current.base_dir, parameters.name, Paths.MODELS_PATH_NAME)
            subconfig_dict, subconfig_path = OvmsConfig.create_subconfig(parameters.name, parameters, config_dir_path_on_host)
            OvmsConfig.replace_subconfig_paths(parameters.name, subconfig_path, resources_dir)

    if parameters.models is not None and any(model.is_mediapipe for model in parameters.models):
        MediaPipeCalculator.prepare_proto_calculator(parameters, config_dir_path_on_host, config_path_on_host)

    cpu_extension_path = None
    if parameters.models is not None and any(isinstance(model, Muse) for model in parameters.models):
        cpu_extension = MuseModelExtension()
        cpu_extension_path = cpu_extension.lib_path[1:]
    elif parameters.cpu_extension:
        if kwargs.get("replace_cpu_extension_params_for_binary", True):
            host_dir = os.path.join(resources_dir, Paths.CPU_EXTENSIONS)
            host_lib_path = os.path.join(host_dir, parameters.cpu_extension.lib_name)
            cpu_extension_path = host_lib_path
            if context.base_os == OsType.Windows:
                raise NotImplementedError("Custom resources are not implemented for Windows")
            copy_custom_lib_to_host(context.ovms_test_image, parameters.cpu_extension.lib_path, host_lib_path)
        else:
            cpu_extension_path = parameters.cpu_extension.lib_path

    if parameters.ports_enabled():
        if context.port_manager_grpc is not None and parameters.grpc_port is None:
            parameters.grpc_port = context.port_manager_grpc.get_port()
        if context.port_manager_rest is not None and parameters.rest_port is None:
            parameters.rest_port = context.port_manager_rest.get_port()

    if parameters.check_version:
        cmd = create_ovms_command(
            config_path=None,
            model_path=None,
            model_name=None,
            parameters=parameters,
            cpu_extension_path=cpu_extension_path,
            batch_size=None,
            shape=None,
            ovms_type=OvmsType.BINARY,
            base_os=context.base_os,
        )
    elif use_config:
        cmd = create_ovms_command(
            config_path=config_path_on_host,
            model_path=None,
            model_name=None,
            parameters=parameters,
            cpu_extension_path=cpu_extension_path,
            batch_size=None,
            shape=None,
            ovms_type=OvmsType.BINARY,
            base_os=context.base_os,
            resolution=parameters.resolution,
        )
    else:
        model = parameters.models[0] if parameters.models is not None else None
        pull = parameters.pull
        task = parameters.task
        task_params = parameters.task_params
        list_models = parameters.list_models
        overwrite_models = parameters.overwrite_models
        add_to_config = parameters.add_to_config
        remove_from_config = parameters.remove_from_config
        batch_size, shape, single_mediapipe_model_mode, gguf_filename = None, None, None, None
        if model is not None and model.is_hf_direct_load:
            source_model = parameters.source_model if parameters.source_model is not None else model.name
            model_repository_path = get_model_repository_path(context, parameters)
            model_name = model.name
            model_path = None
            if model.gguf_filename:
                gguf_filename = model.gguf_filename
        elif list_models is not None:
            model_repository_path = get_model_repository_path(context, parameters)
            source_model, model_name, model_path, gguf_filename = None, None, None, None
        elif add_to_config:
            source_model = None
            model_name = parameters.model_name
            model_path = parameters.model_path
            if model_path is None:
                model_repository_path = get_model_repository_path(context, parameters)
            else:
                model_repository_path = None
            config_path_on_host = os.path.join(TestEnvironment.current.base_dir, context.test_object_name,
                                               Paths.MODELS_PATH_NAME, Paths.CONFIG_FILE_NAME)
        elif remove_from_config:
            source_model, model_repository_path, model_path, gguf_filename = None, None, None, None
            model_name = parameters.model_name
            config_path_on_host = os.path.join(TestEnvironment.current.base_dir, context.test_object_name,
                                               Paths.MODELS_PATH_NAME, Paths.CONFIG_FILE_NAME)
        else:
            model_name = parameters.model_name if parameters.model_name is not None else model.name
            model_path = parameters.model_path if parameters.model_path is not None else model.get_model_path()
            model_path = model_path.replace(Paths.MODELS_PATH_INTERNAL, f"{resources_dir}{Paths.MODELS_PATH_INTERNAL}")
            source_model, model_repository_path, gguf_filename = None, None, None
            batch_size = model.batch_size
            shape = model.input_shape_for_ovms
            single_mediapipe_model_mode = model.single_mediapipe_model_mode
        cmd = create_ovms_command(
            config_path=config_path_on_host,
            model_path=model_path,
            model_name=model_name,
            parameters=parameters,
            cpu_extension_path=cpu_extension_path,
            batch_size=batch_size,
            shape=shape,
            ovms_type=OvmsType.BINARY,
            base_os=context.base_os,
            single_mediapipe_model_mode=single_mediapipe_model_mode,
            pull=pull,
            source_model=source_model,
            gguf_filename=gguf_filename,
            model_repository_path=model_repository_path,
            task=task,
            task_params=task_params,
            list_models=list_models,
            overwrite_models=overwrite_models,
            add_to_config=add_to_config,
            remove_from_config=remove_from_config,
            resolution=parameters.resolution,
            cache_size=parameters.cache_size,
            pooling=model.pooling if model is not None else None,
        )

    ovms_binary = OvmsBinary(
        name=parameters.name if parameters.name is not None else context.test_object_name,
        parameters=parameters,
        cmd=cmd,
        path_to_binary=path_to_binary_ovms,
        container_folder=resources_dir,
    )
    ovms_binary.start(
        base_os=context.base_os,
        environment=environment,
        venv_activate_path=kwargs.get("venv_activate_path", None),
    )
    context.test_objects.append(ovms_binary)

    return OvmsRunContext(ovms_binary, parameters.models)


def get_model_repository_path(context, parameters):
    if parameters.model_repository_path is not None:
        model_repository_path = parameters.model_repository_path
    else:
        model_repository_path = os.path.join(
            TestEnvironment.current.base_dir,
            context.test_object_name,
            Paths.MODELS_PATH_NAME,
        )
        if not os.path.exists(model_repository_path):
            Path(model_repository_path).mkdir(parents=True, exist_ok=True)
    return model_repository_path


class OvmsBinary(OvmsInstance):

    def __init__(self, name, parameters, cmd, path_to_binary, container_folder=None, **kwargs):
        self.process = Process() if kwargs.get("process", None) is None else kwargs.pop("process")
        self.cmd = cmd
        self.path_to_binary = path_to_binary
        super().__init__(
            name=name,
            container_folder=container_folder,
            default_logger=BinaryOvmsLogMonitor(self.process),
            rest_port=parameters.rest_port,
            grpc_port=parameters.grpc_port,
            target_device=parameters.target_device,
            **kwargs,
        )

    def fetch_and_store_ovms_pid(self, timeout=60):
        ovms_pid = None
        parent_proc_pid = self.process._proc.pid
        parent_proc = psutil.Process(parent_proc_pid)
        start = datetime.now()
        while (datetime.now() - start).total_seconds() <= timeout:
            for child in parent_proc.children():
                if "ovms" in child.name():
                    ovms_pid = child.pid
                    break
            if ovms_pid is not None:
                break
        self._dmesg_log.ovms_pid = ovms_pid
        return ovms_pid

    def execute_command(self, cmd, cwd=None, stream=False):
        cwd = TestEnvironment.current.base_dir.strpath if cwd is None else cwd
        proc = Process()
        returncode, stdout, stderr = proc.run(cmd, cwd=cwd)
        assert returncode in [
            0,
            None,
        ], f"Unexpected error code detected: {returncode} (expect 0) during executing command: {cmd}; Error {stderr}"
        return returncode, stdout

    def start(self, ensure_started=False, environment=None, *args, **kwargs):
        self._dmesg_log.get_all_logs()
        base_os = kwargs.get("base_os", None)
        # In order to correctly handle OVMS logs it is required to redirect stderr stream to stdout: 2>&1
        if base_os == OsType.Windows:
            resource_dir = os.path.join(*Path(self.path_to_binary).parts[:-2])
            venv_activate_path = kwargs.get("venv_activate_path", None)
            # full path required for Windows to find the binary
            pre_cmd, env = get_ovms_binary_cmd_setup(
                base_os=base_os,
                resources_dir_path=os.path.join(resource_dir, "ovms"),
                environment=environment,
                venv_activate_path=venv_activate_path,
            )
            cmd = f"{pre_cmd}{resource_dir}\\{self.cmd} 2>&1"
        else:
            resource_dir = os.path.join(*Path(self.path_to_binary).parts[:-3])
            pre_cmd, env = get_ovms_binary_cmd_setup(
                base_os=base_os,
                resources_dir_path=os.path.join(resource_dir, "ovms"),
                environment=environment,
            )
            cmd = f"{pre_cmd} ./{self.cmd} 2>&1"
        self.process.async_run(cmd, cwd=resource_dir, env=env)

    def ensure_status(self, status: str = CONTAINER_STATUS_RUNNING):
        self.process.is_alive()

    def get_status(self, status=None, timeout=None):
        return CONTAINER_STATUS_RUNNING if self.process.is_alive() else CONTAINER_STATUS_EXITED

    def _create_logger(self) -> LogMonitor:
        return BinaryOvmsLogMonitor(self.process)

    def cleanup(self):
        self.kill()
        super().cleanup()
        self.release_ports()
        self.get_dmesg_log_monitor().raise_on_unexpected_messages(filter_known_messages=True)

    def kill(self):
        self.process.kill(force=True) if self.process.is_alive() else True

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
        resources_dir, models_dir_on_host = TestEnvironment.current.prepare_container_folders(name, models)

        if models_to_verify:
            ovms_log = self.create_log(False)

        config_path_on_host = os.path.join(resources_dir, os.path.join(Paths.MODELS_PATH_NAME, Paths.CONFIG_FILE_NAME))
        if models:
            if models_to_verify is not None and any(model.is_mediapipe for model in models_to_verify):
                assert params is not None, "Params should be provided to create MediaPipe calculators"
                self.prepare_mediapipe_config_and_graph(name, params, models)
                OvmsConfig.replace_config_models_paths_for_binary(
                    context, config_path_on_host, resources_dir, name, **kwargs
                )
            else:
                OvmsConfig.generate(name, models)
                OvmsConfig.replace_config_models_paths_for_binary(
                    context, config_path_on_host, resources_dir, name, **kwargs
                )
        else:
            OvmsConfig.generate(name, models)

        config_dict = json.loads(Path(config_path_on_host).read_text())

        if models_to_verify:
            break_msg_list = self.get_break_msg_list(models_to_verify)
            ovms_log.models_loaded(models_to_verify, break_msg_list=break_msg_list)

        return config_dict
