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
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import docker

from tests.functional.utils.context import Context
from tests.functional.utils.inference.communication import REST
from tests.functional.utils.logger import get_logger
from tests.functional.utils.process import Process
from tests.functional.utils.test_framework import generate_test_object_name
from tests.functional.object_model.command_wrappers.ovms import OvmsCommand, create_ovms_command
from tests.functional.constants.core import CONTAINER_STATUS_DEAD, CONTAINER_STATUS_EXITED, CONTAINER_STATUS_RUNNING
from tests.functional.constants.ovms import Config, Ovms
from tests.functional.constants.ovms_type import OvmsType
from tests.functional.constants.paths import Paths
from tests.functional.constants.pipelines import MediaPipe
from tests.functional.constants.target_device_configuration import (
    DEVICES,
    DOCKER_PARAMS,
    HOST,
    NETWORK,
    PRIVILEGED,
    TARGET_DEVICE_CONFIGURATION,
    VOLUMES,
)
from tests.functional.utils.docker import DockerContainer, Limits
from tests.functional.object_model.mediapipe_calculators import MediaPipeCalculator
from tests.functional.object_model.ovms_config import OvmsConfig
from tests.functional.object_model.ovms_instance import OvmsInstance
from tests.functional.object_model.ovms_log_monitor import OvmsCmdLineDockerLogMonitor, OvmsDockerLogMonitor
from tests.functional.object_model.ovms_mapping_config import OvmsMappingConfig
from tests.functional.object_model.ovms_params import OvmsParams
from tests.functional.object_model.ovsa import OvsaCerts
from tests.functional.object_model.test_environment import TestEnvironment
from tests.functional.object_model.ovms_info import OvmsInfo

logger = get_logger(__name__)


@dataclass(frozen=False)
class OvmsDockerParams(OvmsParams):
    detach: bool = True
    limits: Limits = None
    privileged: bool = False
    ovsa_certs: OvsaCerts = None
    process_params: Callable[[dict], None] = None
    volumes: dict = None
    network: str = None


class OvmsDockerLauncher(object):

    @classmethod
    def _update_ports(cls, context, port, parameters, ovms_instance_params, full_cmd):
        if port == parameters.grpc_port:
            parameters.grpc_port = context.port_manager_grpc.get_port()
            updated_port = parameters.grpc_port
            context.port_manager_grpc.release_port(port)
            ovms_instance_params["instance_kwargs"]["grpc_port"] = updated_port
        elif port == parameters.rest_port:
            parameters.rest_port = context.port_manager_rest.get_port()
            updated_port = parameters.rest_port
            context.port_manager_rest.release_port(port)
            ovms_instance_params["instance_kwargs"]["rest_port"] = updated_port
        logger.info(f"Updating ports for OVMS docker. Port {port} was reserved. Trying port {updated_port}")
        ovms_instance_params["command"] = re.sub(rf"{port}", f"{updated_port}", ovms_instance_params["command"])
        del ovms_instance_params["docker_kwargs"]["ports"][f"{port}/tcp"]
        ovms_instance_params["docker_kwargs"]["ports"][f"{updated_port}/tcp"] = updated_port
        ovms_instance_params["name"] = f"{ovms_instance_params['name']}_{updated_port}"
        full_cmd = re.sub(rf"{port}", f"{updated_port}", full_cmd)
        cls._log_create_docker(ovms_instance_params, full_cmd)

    @staticmethod
    def _log_create_docker(ovms_instance_params, full_cmd):
        logger.info(
            f"Docker info\nContainer {ovms_instance_params['name']}\nimage: {ovms_instance_params['image']}\n"
            f"command: {ovms_instance_params['command']}"
        )
        logger.info(f"Docker kwargs: {ovms_instance_params['docker_kwargs']}")
        logger.info(f"Instance kwargs: {ovms_instance_params['instance_kwargs']}")
        logger.info(f"Running cmd: {full_cmd}")

    @classmethod
    def create(
        cls,
        context: Context,
        parameters: OvmsDockerParams,
        ovms_docker_type,
        environment,
        entrypoint=None,
        entrypoint_params=None,
        ovms_instance_params=None,
    ):
        if parameters.models is not None:
            logger.info(
                "Creating ovms with model(s): {}".format(", ".join([model.name for model in parameters.models]))
            )
        if parameters.name is None:
            parameters.name = (
                context.test_object_name if context.test_object_name is not None else generate_test_object_name()
            )
        if parameters.target_device is None and not parameters.single_mediapipe_model_mode:
            # Batch_size and target_device are not supported for single model mode in MediaPipe
            parameters.target_device = context.target_device
        if parameters.image is None:
            parameters.image = context.ovms_image

        if ovms_instance_params is None:
            ovms_instance_params = cls.build_ovms_instance_params(context, parameters)
        if parameters.process_params is not None:
            parameters.process_params(ovms_instance_params)

        if isinstance(ovms_instance_params["command"], OvmsCommand):
            ovms_instance_params["command"] = str(ovms_instance_params["command"])

        full_cmd = parse_cmd(ovms_instance_params, environment, entrypoint, entrypoint_params)

        cls._log_create_docker(ovms_instance_params, full_cmd)

        target_device_lock = OvmsInstance.acquire_target_device_lock(parameters.target_device)

        if ovms_docker_type == OvmsType.DOCKER_CMD_LINE:
            process = Process()
            docker_id = process.run_and_check(full_cmd)
            ovms_docker = OvmsCmdLineDockerInstance(
                docker_id.strip(),
                ovms_instance_params["name"],
                ovms_instance_params["instance_kwargs"]["container_folder"],
                ovms_instance_params["instance_kwargs"]["rest_port"],
                ovms_instance_params["instance_kwargs"]["grpc_port"],
                target_device=ovms_instance_params["instance_kwargs"]["target_device"],
                lock_file=target_device_lock,
            )
        else:
            kwargs = {"environment": environment, "entrypoint": entrypoint}
            if context.terminate_signal_type is not None:
                kwargs["stop_signal"] = context.terminate_signal_type
            container = None
            while container is None:
                try:
                    container = DockerContainer(None).client.run(
                        ovms_instance_params["image"],
                        ovms_instance_params["command"],
                        stdout=True,
                        stderr=False,
                        remove=False,
                        detach=parameters.detach,
                        name=ovms_instance_params["name"],
                        **ovms_instance_params["docker_kwargs"],
                        **kwargs,
                    )
                except docker.errors.APIError as e:
                    match = re.search(r"\d\.\d\.\d\.\d\:(\d{4,5})", e.explanation)
                    if match:
                        # in case port is already allocated choose new port
                        port = int(match.group(1))
                        cls._update_ports(context, port, parameters, ovms_instance_params, full_cmd)
                    else:
                        raise e
            docker_container = DockerContainer(
                container,
                ovms_instance_params["command"],
                parameters.detach,
                **ovms_instance_params["docker_kwargs"],
                **kwargs,
            )
            ovms_docker = OvmsDockerInstance(
                context=context,
                container=docker_container,
                lock_file=target_device_lock,
                **ovms_instance_params["instance_kwargs"],
            )

        ovms_docker._dmesg_log.get_all_logs()
        context.test_objects.append(ovms_docker)
        return ovms_docker

    @classmethod
    def _prepare_ports(cls, grpc_port: int = None, rest_port: int = None) -> dict:
        ports = dict()
        if grpc_port is not None:
            ports.update({f"{grpc_port}/tcp": grpc_port})
        if rest_port is not None:
            ports.update({f"{rest_port}/tcp": rest_port})
        return ports

    @classmethod
    def prepare_models_mapping(cls, context, ovms_container, models):
        if not models:
            return
        for model in models:
            mapping_exists = OvmsMappingConfig.mapping_exists(ovms_container, model)
            if model.use_mapping is None:
                continue  # Use default mapping.json (if exists) do not touch nor modify.
            if model.use_mapping is True:
                if mapping_exists:
                    # Delete original mapping since it is tested in case: `default_model_mapping`
                    OvmsMappingConfig.delete_mapping(model)
                OvmsMappingConfig.generate(model, context)  # create generic mapping
            if model.use_mapping is False:
                if mapping_exists:
                    OvmsMappingConfig.delete_mapping(model)  # just delete mapping

    @classmethod
    def build_ovms_instance_params(cls, context: Context, parameters: OvmsDockerParams):
        name = parameters.name
        regular_models = parameters.get_regular_models()
        container_folder, models_dir_on_host = TestEnvironment.current.prepare_container_folders(
            name, parameters.get_models()
        )
        volumes = parameters.volumes if parameters.volumes is not None else \
            cls.prepare_new_volumes_for_container(models_dir_on_host)
        if parameters.use_cache:
            if parameters.cache_dir_path is None:
                cache_dir_on_host = os.path.join(container_folder, "../cache")
            else:
                cache_dir_on_host = parameters.cache_dir_path
            cls.prepare_volume_for_cache(cache_dir_on_host, volumes)

        if parameters.allowed_local_media_path is not None:
            cls.prepare_volume_for_images(parameters.allowed_local_media_path, volumes)

        cls.prepare_models_mapping(context, container_folder, parameters.models)

        config_path_on_host, config_path = cls.create_config(parameters, name)

        config_file = None
        config_data = ""
        if config_path_on_host is not None:
            if parameters.volumes is None:
                volumes.update(cls.prepare_new_volumes_for_container([config_path_on_host]))
            config_file = os.path.join(config_path_on_host, Paths.CONFIG_FILE_NAME)
            config_data = Path(config_file).read_text()

        # Create and save .pbtxt file for each model in pipeline
        if (
            parameters.custom_config is not None
            and parameters.custom_config.get(Config.MEDIAPIPE_CONFIG_LIST) is not None
            or Config.MEDIAPIPE_CONFIG_LIST in config_data
            or "mediapipe" in config_data
            or (not parameters.use_config and parameters.single_mediapipe_model_mode)
        ):
            MediaPipeCalculator.prepare_proto_calculator(parameters, config_path_on_host, config_file)

        assert not (
            parameters.shape and len(regular_models) > 1
        ), "Forbidden to pass more than one model and `shape` or `batch_size` parameters."

        cpu_extension_path = None
        if parameters.cpu_extension:
            cpu_extension_path = parameters.cpu_extension.lib_path

        OvmsInfo.get_local_image(parameters.image)
        image = OvmsInfo.IMAGES[parameters.image]

        target_device_is_valid = parameters.target_device in TARGET_DEVICE_CONFIGURATION

        extra_docker_params = {}
        batch_size, shape = None, None
        model_name, model_path = parameters.model_name, parameters.model_path
        pull = parameters.pull
        task = parameters.task
        task_params = parameters.task_params
        list_models = parameters.list_models
        overwrite_models = parameters.overwrite_models
        add_to_config = parameters.add_to_config
        remove_from_config = parameters.remove_from_config
        resolution = parameters.resolution
        cache_size = parameters.cache_size
        source_model, model_repository_path, gguf_filename = None, None, None
        extra_docker_params.update({"user": f"{os.getuid()}:{os.getgid()}"})
        model = None
        if config_path is None and regular_models:
            model = regular_models[0]
            if model.is_hf_direct_load:
                source_model = parameters.source_model if parameters.source_model is not None else model.name
                model_repository_path = parameters.model_repository_path \
                    if parameters.model_repository_path is not None else Paths.MODELS_PATH_INTERNAL
                if model.gguf_filename:
                    gguf_filename = model.gguf_filename
                model_name = model.name
                model_path = None
                volumes = parameters.volumes if parameters.volumes is not None else \
                    cls.prepare_new_volumes_for_container(models_dir_on_host, mode="rw")
            else:
                # Batch_size and target_device are not supported for single model mode in MediaPipe
                batch_size = model.batch_size if not parameters.single_mediapipe_model_mode else None
                shape = model.input_shape_for_ovms
                if model_name is None:
                    model_name = model.name
                if model_path is None:
                    model_path = model.get_model_path()
        elif list_models is not None:
            model_repository_path = parameters.model_repository_path \
                if parameters.model_repository_path is not None else Paths.MODELS_PATH_INTERNAL
            source_model, model_name, model_path = None, None, None
        elif add_to_config:
            source_model = None
            model_name = parameters.model_name
            model_path = parameters.model_path
            if model_path is None:
                model_repository_path = parameters.model_repository_path \
                    if parameters.model_repository_path is not None else Paths.MODELS_PATH_INTERNAL
            else:
                model_repository_path = None
            config_path = Paths.CONFIG_PATH_INTERNAL
        elif remove_from_config:
            source_model, model_repository_path, model_path = None, None, None
            model_name = parameters.model_name
            config_path = Paths.CONFIG_PATH_INTERNAL

        ports = cls._prepare_ports(parameters.grpc_port, parameters.rest_port)
        if target_device_is_valid and not parameters.check_version:
            # Note: currently we use lambda: expression for performing 'lazy init' of syscalls:
            # getuid() & getgrnam('users')/getgrnam('render')
            target_device_conf = TARGET_DEVICE_CONFIGURATION[parameters.target_device]()
            devices = target_device_conf[DEVICES]
            network = parameters.network if parameters.network is not None else target_device_conf[NETWORK]
            if DOCKER_PARAMS in target_device_conf:
                extra_docker_params.update(target_device_conf[DOCKER_PARAMS])
            privileged = parameters.privileged or target_device_conf[PRIVILEGED]
            volumes = cls.update_volume_for_mounts(target_device_conf[VOLUMES], volumes)
        else:  # required for negative test cases
            devices = []
            network = parameters.network
            privileged = parameters.privileged

        ovsa_certs = parameters.ovsa_certs if parameters.ovsa_certs is not None else OvsaCerts.default_certs
        if ovsa_certs is not None:
            volumes.update(ovsa_certs.create_ovsa_volume_bindings())
        if parameters.custom_command is None:
            command = create_ovms_command(
                config_path=config_path,
                model_path=model_path,
                model_name=model_name,
                parameters=parameters,
                cpu_extension_path=cpu_extension_path,
                batch_size=batch_size,
                shape=shape,
                ovms_type=OvmsType.DOCKER,
                base_os=context.base_os,
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
                resolution=resolution,
                cache_size=cache_size,
                pooling=model.pooling if model is not None else None,
            )
        else:
            command = parameters.custom_command

        docker_kwargs = dict(volumes=volumes, devices=devices, network=network, privileged=privileged)
        docker_kwargs.update(extra_docker_params)

        instance_kwargs = dict(
            container_folder=container_folder,
            rest_port=parameters.rest_port,
            grpc_port=parameters.grpc_port,
            target_device=parameters.target_device,
        )
        if parameters.limits:
            docker_kwargs.update(parameters.limits)

        if network != HOST:
            # this sends ports to docker api create/run methods
            docker_kwargs["ports"] = ports

        result = {
            "image": image,
            "command": command,
            "name": name,
            "ports": "ports",
            "docker_kwargs": docker_kwargs,
            "instance_kwargs": instance_kwargs,
        }
        return result

    @classmethod
    def create_config(cls, parameters, name):
        regular_models = parameters.get_regular_models()
        using_custom_loader = any([(x.custom_loader is not None) for x in regular_models])
        if using_custom_loader and not parameters.use_config:
            msg = f"Custom loader is supported only with config file passed with --config_path."
            logger.error(msg)
            raise Exception(msg)

        config_dir_path_on_host = os.path.join(TestEnvironment.current.base_dir, name, Paths.MODELS_PATH_NAME)
        if parameters.create_config_method is not None:
            parameters.create_config_method(os.path.join(config_dir_path_on_host, Paths.CONFIG_FILE_NAME))
            use_config = True
        elif parameters.custom_config is not None:
            OvmsConfig.save(name, parameters.custom_config)
            use_config = True
        else:
            # Automatically set use_config if the following conditions are met
            use_config = (
                len(regular_models) > 1
                or parameters.use_config
                or any((isinstance(model, MediaPipe) and not model.single_mediapipe_model_mode)
                        for model in regular_models)
            )
            if use_config:
                OvmsConfig.generate_from_parameters(name, parameters)

        if parameters.use_subconfig:
            OvmsConfig.create_subconfig(name, parameters, config_dir_path_on_host)

        if use_config:
            return config_dir_path_on_host, Paths.CONFIG_PATH_INTERNAL
        else:
            return None, None

    @staticmethod
    def prepare_new_volumes_for_container(container_folders, mode="ro"):
        """
        Prepare dictionary with instruction how docker should map those locations.

        Parameters:
        container_folders (set(str)): Set of path located on host that should be mapped to docker.

        Returns:
        dict: Instruction how to map host directories to docker instance.
        """
        result = {}
        for resource in container_folders:
            base_dir = os.path.basename(resource)
            if "_mediapipe" in base_dir:
                # Single model mode in MediaPipe
                result[resource] = {
                    "bind": os.path.join(Paths.OVMS_PATH_INTERNAL, Paths.MODELS_PATH_NAME, base_dir), "mode": mode,
                }
            else:
                result[resource] = {"bind": os.path.join(Paths.OVMS_PATH_INTERNAL, base_dir), "mode": mode}

        return result

    @staticmethod
    def prepare_volume_for_cache(cache_dir_on_host, volumes):
        oldmask = os.umask(0)
        try:
            os.makedirs(cache_dir_on_host, mode=0o777, exist_ok=True)
        finally:
            os.umask(oldmask)
        volumes.update({cache_dir_on_host: {"bind": Paths.CACHE_INTERNAL, "mode": "rw"}})

    @staticmethod
    def prepare_volume_for_images(images_path, volumes):
        volumes.update({images_path: {"bind": Paths.IMAGES_PATH_INTERNAL, "mode": "rw"}})

    @staticmethod
    def update_volume_for_mounts(mounts: list, volumes: dict):
        for mount in mounts:
            volumes.update(mount)
        return volumes


def parse_cmd(result, environment, entrypoint, entrypoint_params):

    full_cmd = f"docker run -d"
    if result["docker_kwargs"]["privileged"]:
        full_cmd += " --privileged"

    if result["docker_kwargs"]["network"]:
        full_cmd += f" --network {result['docker_kwargs']['network']}"

    if result["docker_kwargs"].get("user", None):
        full_cmd += f" --user {result['docker_kwargs']['user']}"

    for cgroup_rule in result["docker_kwargs"].get("device_cgroup_rules", []):
        full_cmd += f" --device-cgroup-rule '{cgroup_rule}'"

    for device in result["docker_kwargs"]["devices"]:
        full_cmd += f" --device {device}"

    for group_add in result["docker_kwargs"].get("group_add", []):
        full_cmd += f" --group-add {group_add}"

    for container_port, host_port in result["docker_kwargs"]["ports"].items():
        port = container_port.partition("/")[0]
        full_cmd += f" -p {host_port}:{port}"

    env_str = ""
    if environment:
        for key, value in environment.items():
            env_str += f" -e {key}='{value}'"

    full_cmd += env_str

    for volume, bind_info in result["docker_kwargs"]["volumes"].items():
        full_cmd += f" -v {volume}:{bind_info['bind']}:{bind_info['mode']}"

    if entrypoint is not None:
        full_cmd += f" --entrypoint {entrypoint}"

    full_cmd += " " + result["image"].tags[0]

    if entrypoint_params is not None:
        result["command"] = f"{entrypoint_params} {result['command']}"

    # '/ovms/bin/ovms --log_level INFO --port 9007 --rest_port 8005 --config_path /models/config.json'
    if entrypoint is not None:
        full_cmd += " " + result["command"] if type(result["command"]) == str else " " + " ".join(result["command"])
    elif "/ovms/bin/ovms" in result["command"]:
        result["command"] = result["command"].partition("/ovms/bin/ovms")[2].strip()
        full_cmd += " " + result["command"]
    else:
        full_cmd += " " + " ".join(result["command"])
    return full_cmd


class OvmsDockerInstance(OvmsInstance):

    def __init__(self, container, container_folder, rest_port, grpc_port, target_device, **kwargs):
        if container is not None:
            self.container = container
            self.name = container.container.name
        super().__init__(
            name=self.name,
            container_folder=container_folder,
            default_logger=OvmsDockerLogMonitor(container.container),
            rest_port=rest_port,
            grpc_port=grpc_port,
            target_device=target_device,
            **kwargs,
        )

    def fetch_and_store_ovms_pid(self, timeout=60):
        docker_client = docker.APIClient()
        docker_inspect_output = docker_client.inspect_container(self.get_short_id())
        self.ovms_pid = docker_inspect_output["State"]["Pid"]
        super().fetch_and_store_ovms_pid()
        logger.info(f"OVMS Process ID = [{self.ovms_pid}]")
        return self.ovms_pid

    def is_ovms_running(self):
        return self.get_status() not in [CONTAINER_STATUS_EXITED, CONTAINER_STATUS_DEAD]

    def get_ip(self):
        return self.container.container.attrs["NetworkSettings"]["IPAddress"]

    def get_rest_port(self):
        return self.ovms_ports[REST]

    def get_status(self, status=None, timeout=None):
        return self.container.get_status()

    def _create_logger(self):
        return OvmsDockerLogMonitor(self.container.container)

    def execute_command(self, cmd, stream=False, cwd=None, workdir=None):
        exit_code, output = self.container.container.exec_run(user="root", cmd=cmd, stream=stream, workdir=workdir)
        stdout = output if stream else output.decode()
        return exit_code, stdout

    def get_short_id(self):
        return self.container.container.short_id

    def cleanup(self):
        if not self.container.deleted:
            try:
                super().cleanup()
            except ValueError as e:
                print(f"Error occurred during cleanup:\n{e}")
            finally:
                try:
                    logger.info(f"Removing container {self.container.container.short_id}")
                    self.container.cleanup()
                    logger.info("Container removed")
                    self.release_ports()
                    self._dmesg_log.raise_on_unexpected_messages(filter_known_messages=True)
                except Exception as e:
                    logger.info(f"Cleanup triggered exception: {str(e)}")
                    raise e

    def start(self, ensure_started=False, *args, **kwargs):
        self.container.start_container()
        if ensure_started:
            self.ensure_started(*args, **kwargs)

    def stop_ovms(
            self, context, ensure_deleted=False, terminate_signal_type=Ovms.SIGTERM_SIGNAL, remove_container=True
    ):
        context.terminate_signal_type = (
            terminate_signal_type if context.terminate_signal_type is None else context.terminate_signal_type
        )
        signal = self.get_signal_type(context.terminate_signal_type)
        if context.terminate_method == Ovms.STOP_METHOD:
            self.container.stop_container()
        else:
            self.container.kill_container(signal=signal)
        if remove_container:
            self.container.remove_container(ensure_deleted)
            self.container._set_deleted(True)

    @staticmethod
    def from_docker_id(docker_id, **kwargs):
        client = docker.from_env()
        container = client.containers.get(docker_id)
        docker_container = DockerContainer(container)
        return OvmsDockerInstance(docker_container, "", None, None, None, **kwargs)


class OvmsCmdLineDockerInstance(OvmsInstance):

    def __init__(self, docker_id, name, container_folder, rest_port, grpc_port, target_device, **kwargs):
        self.docker_id = docker_id
        self._disposed = False
        super().__init__(
            name=name,
            container_folder=container_folder,
            default_logger=OvmsCmdLineDockerLogMonitor(self.docker_id),
            rest_port=rest_port,
            grpc_port=grpc_port,
            target_device=target_device,
            **kwargs,
        )

    def fetch_and_store_ovms_pid(self, timeout=60):
        self._dmesg_log.ovms_pid = None  # Not implemetned yet

    def _create_logger(self):
        return OvmsCmdLineDockerLogMonitor(self.docker_id)

    def ensure_status(self, status: str = CONTAINER_STATUS_RUNNING, timeout: int = 60):
        process = Process()
        short_id = self.get_short_id()
        timeout = time.time() + timeout
        current_status = None
        while True:
            try:
                _, _stdout, _ = process.run(f"docker ps --filter id={short_id}")
                if short_id in _stdout:
                    current_status = CONTAINER_STATUS_RUNNING
                else:
                    current_status = CONTAINER_STATUS_EXITED
                if current_status == status:
                    break
            except AssertionError:
                pass
            if time.time() > timeout:
                raise TimeoutError(f"Current status: {current_status}")
        return current_status

    def get_status(self, status=None, timeout=60):
        return self.ensure_status(status, timeout=timeout)

    def stop_container(self, signal=Ovms.SIGTERM_SIGNAL):
        process = Process()
        process.run(f"docker stop --signal {signal} {self.docker_id}")

    def kill_container(self, signal=Ovms.SIGKILL_SIGNAL):
        process = Process()
        process.run(f"docker kill --signal {signal} {self.docker_id}")  # SIGTERM, SIGINT; default: SIGKILL

    def remove_container(self, ensure_deleted: bool = False):
        process = Process()
        process.run(f"docker rm {self.docker_id}")
        if ensure_deleted:
            timeout = time.time() + 60
            while True:
                short_id = self.get_short_id()
                _, stdout, _ = process.run(f"docker ps --filter id={self.get_short_id()}")
                if short_id not in stdout:
                    break
                elif time.time() > timeout:
                    raise TimeoutError(f"Container {short_id} is not removed")

    def execute_command(self, cmd, stream=False, cwd=None):
        process = Process()
        if stream:
            detach = "-d"
        else:
            detach = ""
        exit_code, stdout, stderr = process.run(f"docker exec {detach} -u root {self.docker_id} {cmd}", cwd=cwd)
        return exit_code, stdout

    def get_env_variables(self):
        _, printenv_output = self.execute_command("env")
        output = {}
        if printenv_output:
            for line in printenv_output.split():
                env_variable = line.split("=")
                output[env_variable[0]] = env_variable[1]
        return output

    def get_short_id(self):
        return self.docker_id[0:11]

    def cleanup(self):
        if not self._disposed:
            try:
                super().cleanup()
            finally:
                try:
                    process = Process()
                    process.run_and_check(f"docker kill {self.docker_id}")
                    self.release_ports()
                except Exception as e:
                    logger.info(str(e))

                self._disposed = True

    def start(self, ensure_started=False, *args, **kwargs):
        pass

    def stop_ovms(self, context, ensure_deleted=False):
        signal = self.get_signal_type(context.terminate_signal_type)
        if context.terminate_method == Ovms.STOP_METHOD:
            self.stop_container(signal=signal)
        else:
            self.kill_container(signal=signal)
        self.remove_container(ensure_deleted)
