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

import pprint
import signal
import time
from abc import ABCMeta
from io import BytesIO
from typing import Any, Callable, List, Tuple, Union

import docker
from docker import Context
from docker.models.containers import Container
from retry.api import retry_call
from typing_extensions import TypedDict

from tests.functional.utils.logger import get_logger
from tests.functional.utils.process import Process
from tests.functional.utils.test_framework import generate_test_object_name
from tests.functional.config import docker_client_timeout
from tests.functional.constants.core import CONTAINER_STATUS_RUNNING

logger = get_logger(__name__)


class DockerClient(docker.DockerClient):

    def build(self, dockerfile: str, build_args, nocache: bool = True, **kwargs) -> tuple:
        logs = []
        with open(dockerfile, "r") as file:
            data = file.read()
        file_obj = BytesIO(data.encode("utf-8"))
        image, generator = self.images.build(fileobj=file_obj, nocache=nocache, buildargs=build_args, **kwargs)
        while True:
            try:
                output = generator.__next__()
                logs.append(str(output.values()))
            except StopIteration:
                break

        return image, logs

    def push(self, repository, tag=None, **kwargs):
        logs = self.images.push(repository=repository, tag=tag, **kwargs).split("\n")

        for line in logs:
            assert (
                "requested access to the resource is denied" not in line
            ), "Unauthorized to push docker image: {}".format(line)
            assert "error" not in line, "Failed to push docker image: {}".format(line)
        return logs

    def pull(self, repository, tag):
        return self.images.pull(repository=repository, tag=tag)

    def remove(self, image_id: str, **kwargs):
        return self.images.remove(image=image_id, **kwargs)

    def get(self, container_id_or_name) -> Container:
        container = self.containers.get(container_id_or_name)
        return container

    def create(self, image, command=None, **kwargs) -> Container:
        container = self.containers.create(image, command, **kwargs)
        return container

    def run(self, image, command=None, stdout=True, stderr=False, remove=False, **kwargs) -> Union[Container, None]:
        container = self.containers.run(image, command, stdout, stderr, remove, **kwargs)
        return container

    def list_containers(self, all_containers=False, before=None, filters=None, limit=-1, since=None) -> List[Container]:
        return self.containers.list(all_containers, before, filters, limit, since)


class Limits(TypedDict):
    cpu_period: int
    cpu_quota: int
    cpu_shares: int
    cpuset_cpus: str
    kernel_memory: Union[int, str]
    mem_limit: Union[int, str]
    mem_reservation: Union[int, str]
    mem_swappiness: int
    memswap_limit: Union[int, str]
    oom_kill_disable: bool


class DockerContainer(metaclass=ABCMeta):
    TITLES_HEADER = "Titles"
    PROCESSES_HEADER = "Processes"
    PID_HEADER = "PID"
    COMMAND_HEADER = "CMD"
    COMMON_RETRY = {"tries": 30, "delay": 2}
    NOT_ON_LIST_RETRY = {"tries": 10, "delay": 2}
    GETTING_LOGS_RETRY = COMMON_RETRY
    GETTING_STATUS_RETRY = COMMON_RETRY

    def __init__(
        self,
        container: Container,
        command=None,
        detach: bool = True,
        ports: dict = None,
        volumes: dict = None,
        devices: List[type(str)] = None,
        limits: Limits = None,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.container = container
        self.image = None if not container else self.container.image
        self.name = None if not container else self.container.name
        self.command = command
        self.detach = detach
        self.ports = ports
        self.volumes = volumes
        self.limits = limits
        self.devices = devices
        self.id = self.name
        self._is_deleted = False
        self.client = DockerClient(timeout=docker_client_timeout)

    @classmethod
    def run(
        cls,
        context: Context,
        image,
        command=None,
        stdout=True,
        stderr=False,
        remove=False,
        name: str = None,
        detach: bool = True,
        ports: dict = None,
        volumes: dict = None,
        devices: List[type(str)] = None,
        limits: Limits = None,
        **kwargs,
    ):
        logger.info("Running container with:\n image: {}\n command: {}\n volumes: {}".format(image, command, volumes))
        if limits is not None:
            kwargs.update(limits)
        container = cls.client.run(
            image,
            command,
            stdout=stdout,
            stderr=stderr,
            remove=remove,
            name=cls.container_name(name),
            detach=detach,
            ports=ports,
            volumes=volumes,
            devices=devices,
            **kwargs,
        )
        instance = cls(
            container, command, detach=detach, ports=ports, volumes=volumes, devices=devices, limits=limits, **kwargs
        )
        context.test_objects.append(instance)
        return instance

    @classmethod
    def create(
        cls,
        context: Context,
        image,
        command=None,
        name: str = None,
        detach: bool = True,
        ports: dict = None,
        volumes: dict = None,
        devices: List[type(str)] = None,
        limits: Limits = None,
        **kwargs,
    ):
        logger.info("Creating container with:\n image: {}\n command: {}\n volumes: {}".format(image, command, volumes))
        if limits is not None:
            kwargs.update(limits)
        container = cls.client.create(
            image,
            command,
            name=cls.container_name(name),
            detach=detach,
            ports=ports,
            volumes=volumes,
            devices=devices,
            **kwargs,
        )
        instance = cls(
            container, command, detach=detach, ports=ports, volumes=volumes, devices=devices, limits=limits, **kwargs
        )
        context.test_objects.append(instance)
        return instance

    @classmethod
    def get(cls, container_id_or_name) -> Union["DockerContainer", None]:
        container = cls.client.get(container_id_or_name)  # type: Container
        return cls.from_response(container)

    @classmethod
    def list(cls, all_containers=False, before=None, filters=None, limit=-1, since=None) -> List["DockerContainer"]:
        container_list = cls.client.list_containers(
            all_containers, before, filters, limit, since
        )  # type: List[Container]
        return cls.list_from_response(container_list)

    @classmethod
    def from_response(cls, rsp: Container):
        return cls(container=rsp)

    def restart(self, stdout=True, stderr=False, remove=False, **kwargs):
        if len(kwargs):
            self.kwargs.update(kwargs)
        self.delete()
        self.container = self.client.run(
            self.image,
            self.command,
            stdout=stdout,
            stderr=stderr,
            remove=remove,
            name=self.name,
            detach=self.detach,
            ports=self.ports,
            volumes=self.volumes,
            **self.kwargs,
        )

    @classmethod
    def container_name(cls, container_name: str = None):
        return generate_test_object_name() if container_name is None else container_name

    @classmethod
    def volume(cls, external_path: str, internal_path: str, mode: str = "ro", volumes: dict = None):
        if not isinstance(volumes, dict):
            volumes = dict()
        volumes[external_path] = {"bind": internal_path, "mode": mode}
        return volumes

    def start_container(self):
        assert self.container is not None, "Lack of container {} to start (is None)\nContainers found:\n{}".format(
            self.name, repr(self.client.list_containers(all_containers=True))
        )
        return self.container.start()

    def stop_container(self, **kwargs):
        assert self.container is not None, "Lack of container {} to stop (is None)\nContainers found:\n{}".format(
            self.name, repr(self.client.list_containers(all_containers=True))
        )
        return self.container.stop(**kwargs)

    def kill_container(self, signal=signal.SIGTERM):
        assert self.container is not None, "Lack of container {} to kill (is None)\nContainers found:\n{}".format(
            self.name, repr(self.client.list_containers(all_containers=True))
        )
        return self.container.kill(signal=signal)  # SIGKILL (not supported for Windows), SIGINT; default: SIGTERM

    def remove_container(self, ensure_deleted: bool = False):
        assert self.container is not None, "Lack of container {} to remove (is None)\nContainers found:\n{}".format(
            self.name, repr(self.client.list_containers(all_containers=True))
        )
        removed = self.container.remove()
        if ensure_deleted:
            self.ensure_not_on_list(self.name)
        return removed

    def delete(self, ensure_deleted: bool = False):
        self.stop_container()
        self.remove_container(ensure_deleted)

    def check_non_empty_logs(self, specific_str: str, acceptable_logs_length_trigger: int = 0, **kwargs):
        logs = self.get_logs(**kwargs)
        assert len(logs) > acceptable_logs_length_trigger, "Logs list for {} should not be empty".format(self.name)
        assert specific_str in logs, "Specific string: {} not found in logs: {}".format(specific_str, logs)
        return logs

    def ensure_logs_contain_specific_str(
        self, specific_str: str, acceptable_logs_length_trigger: int = 0, retry_kwargs: dict = None, **kwargs
    ):
        args = [specific_str, acceptable_logs_length_trigger]
        getting_logs_retry = self.GETTING_LOGS_RETRY.copy()
        if retry_kwargs:
            getting_logs_retry.update(retry_kwargs)
        return retry_call(
            self.check_non_empty_logs, fargs=args, fkwargs=kwargs, exceptions=AssertionError, **getting_logs_retry
        )

    @property
    def ports_mapping(self):
        self.container.reload()
        return self.container.ports

    def get_first_host_port_mapping(self, mapped_port):
        port_mapping = self.ports_mapping.get(mapped_port, []) or []
        first_host_port_mapping = next(iter(port_mapping), {})
        return first_host_port_mapping

    def get_host_port_mapping(self, mapped_port: str) -> Tuple[str, str]:
        """
        gets first host and port mapping
        :param mapped_port: str -> port mapping in format: "3456/tcp"
        :return: Tuple[str, str] -> (host_ip, host_port)
        """
        host = self.get_first_host_port_mapping(mapped_port)
        host_port = host.get("HostPort", None)
        host_ip = host.get("HostIP", None)
        assert host_port is not None, f"Cannot get mapped port for {mapped_port} in {self.ports}"
        return host_ip, host_port

    def update(self):
        self.container = self.client.containers.get(self.container.id)  # type: Container
        return self

    def get_status(self, status=None, timeout=None):
        self.update()
        return self.container.status

    def assert_status(self, status):
        current_status = self.get_status()
        assert current_status == status, (
            "Not expected status for container {} found. \n "
            "Expected: {}, \n "
            "received: {}".format(self.container.name, status, self.container.status)
        )
        return True

    def ensure_status(self, status: str = CONTAINER_STATUS_RUNNING):
        container_status = {"status": status}
        return retry_call(
            self.assert_status, fkwargs=container_status, exceptions=AssertionError, **self.GETTING_STATUS_RETRY
        )

    def get_logs(self, **kwargs) -> Union[bool, str]:
        assert self.container is not None, "Lack of container to get logs from (is None)"
        return self.container.logs(**kwargs).decode()

    @classmethod
    def check_not_on_list(cls, container: Union[str, "DockerContainer"], comparator: Callable[[Any, Any], bool] = None):
        current_list = cls.list()
        logger.debug(
            "Searching for container with a name: {name}, among:\n{elem}\n".format(
                name=container if isinstance(container, str) else container.name,
                elem="\n".join([repr(elem) for elem in current_list]),
            )
        )
        if comparator is None:
            assert container not in current_list, "{} was found on: {}".format(container, pprint.pformat(current_list))
        else:
            for member in current_list:
                assert comparator(container, member) is False, "{} was found on: {}".format(
                    container, pprint.pformat(current_list)
                )

    @classmethod
    def ensure_not_on_list(
        cls,
        container: Union[str, "DockerContainer"],
        comparator: Callable[[Any, Any], bool] = None,
        ensure_count: int = 1,
    ):
        retry_call(
            cls.check_not_on_list,
            fargs=[container, comparator],
            exceptions=AssertionError,
            tries=cls.NOT_ON_LIST_RETRY["tries"],
            delay=cls.NOT_ON_LIST_RETRY["delay"],
        )
        for count in range(1, ensure_count):
            time.sleep(cls.NOT_ON_LIST_RETRY["delay"])
            cls.check_not_on_list(container, comparator)

    def __repr__(self):
        _id = self.container.id if self.container is not None else ""
        ports = pprint.pformat(self.container.ports) if self.container is not None else "<empty>"
        return "<%s: %s%s>@%s" % (
            self.__class__.__name__,
            self.id,
            " (%s)" % _id,
            "ports: %s." % ports,
        )

    @classmethod
    def list_from_response(cls, rsp):
        """Will create list of object from the response"""
        items = []
        for item in rsp:
            items.append(cls.from_response(item))
        return items

    @property
    def deleted(self):
        return self._is_deleted

    def cleanup(self):
        """Method called by context after tests have finished"""
        self.delete()
        self._set_deleted(True)

    def _set_deleted(self, is_deleted):
        self._is_deleted = bool(is_deleted)

    def list_containers(self):
        containers = self.client.list_containers()
        return containers

    def prune(self, filters=None):
        return self.client.containers.prune(filters)


class DockerNetwork:

    def __init__(self, context, network_name=None):
        self.context = context
        self.network_name = network_name if network_name is not None else context.test_object_name
        self.proc = Process()
        self.proc.disable_check_stderr()
        self.context.test_objects.append(self.cleanup)

    def create_network(self):
        self.proc.run_and_check(f"docker network create {self.network_name}")

    def connect_network(self, container_name):
        self.proc.run(f"docker network connect {self.network_name} {container_name}")

    def disconnect_network(self, container_name):
        self.proc.run(f"docker network disconnect {self.network_name} {container_name}")

    def remove_network(self):
        self.proc.run_and_check(f"docker network rm {self.network_name}")

    def cleanup(self):
        self.remove_network()
