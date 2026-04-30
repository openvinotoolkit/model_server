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

import docker
import re

from tests.functional.utils.environment_info import DEFAULT_FULL_VERSION_NUMBER, BaseInfo
from tests.functional.utils.logger import get_logger
from tests.functional.constants.os_type import get_host_os_details
from tests.functional.utils.process import Process

from tests.functional.config import airplane_mode, base_os, ovms_image_local
from tests.functional.config import tmp_dir
from tests.functional.constants.ovms_binaries import get_binaries, get_ovms_binary_cmd_setup
from tests.functional.constants.ovms_type import OvmsType, OVMS_BINARY_PACKAGE_EXTENSIONS

logger = get_logger(__name__)


class OvmsInfo(BaseInfo):
    """Retrieves OVMS version and os distname from container."""

    _os_distname = None
    _ovms_version = None
    _ov_version = None
    _ov_genai_version = None
    _info_read = False
    _docker_ovms_types = [OvmsType.DOCKER, OvmsType.DOCKER_CMD_LINE, OvmsType.KUBERNETES]

    IMAGES = {}

    @property
    def os_distname(self):
        """This method gets operating system distribution name from given container/binary name."""
        try:
            if self.image is not None and self.image.endswith(OVMS_BINARY_PACKAGE_EXTENSIONS):
                self._get_info_from_binary()
            else:
                self._get_info_from_container()
            return self._os_distname
        except Exception as exc:
            logger.error(f"Couldn't retrieve OVMS os distname from.\nException: {exc}")
            raise exc

    @property
    def version(self):
        """This method gets OVMS version from given container/binary name."""
        try:
            if self.image is not None and self.image.endswith(OVMS_BINARY_PACKAGE_EXTENSIONS):
                self._get_info_from_binary()
            else:
                self._get_info_from_container()
            return f"{self._ovms_version}_OV{self._ov_version}_genAI{self._ov_genai_version}"
        except Exception as exc:
            logger.error(f"Couldn't retrieve OVMS version from.\nException: {exc}")
            raise exc

    def get_ovms_version(self, image=None):
        try:
            if image is not None and image.endswith(OVMS_BINARY_PACKAGE_EXTENSIONS):
                self._get_info_from_binary()
            else:
                self._get_info_from_container(image=image)
            return self._ovms_version
        except Exception as exc:
            logger.error(f"Couldn't retrieve OVMS version from.\nException: {exc}")
            raise exc

    def get_ovms_ov_version(self, image=None):
        try:
            if image is not None and image.endswith(OVMS_BINARY_PACKAGE_EXTENSIONS):
                self._get_info_from_binary()
            else:
                self._get_info_from_container(image=image)
            return self._ovms_version, self._ov_version, self._ov_genai_version
        except Exception as exc:
            logger.error(f"Couldn't retrieve OVMS version from.\nException: {exc}")
            raise exc

    @classmethod
    def get(cls):
        """This method gets OVMS version in dict format."""

        try:
            cls._get_info_from_container()
            return {"version": cls.version()}
        except Exception as exc:
            logger.warning(f"Couldn't retrieve OVMS version from.\nException: {exc}")
            return {"version": DEFAULT_FULL_VERSION_NUMBER}

    def _get_info_from_container(self, image=None):
        """Run container and get all desired information."""
        from tests.functional.utils.docker import DockerClient  # pylint: disable=import-outside-toplevel

        if self._info_read:
            return

        if airplane_mode or ovms_image_local or image is not None:
            image = image if image is not None else self.image
            self.get_local_image(image)
        else:
            image = self.image
            self.docker_pull_image_cli(image)

        client = DockerClient()
        ovms_container = client.create(
            image=image,
            entrypoint="sleep",
            command="inf",
        )

        try:
            ovms_container.start()
            cmd = "/ovms/bin/ovms --version"
            exit_code, version_output = ovms_container.exec_run(cmd=cmd)
            logger.info(f"Container version output: {version_output}")
            assert all([exit_code == 0, "OpenVINO Model Server" in str(version_output)]), (
                f"Failed to run cmd: {cmd}; " f"exit_code: {exit_code}; " f"output: {version_output}"
            )
            _ovms_version_string = version_output.decode().strip()

            self._ovms_version = re.search(r"Model Server\s+(\d[^\r\n]+)", _ovms_version_string).group(1).strip()
            self._ov_version = re.search(r"OpenVINO backend\s+([^\r\n]+)", _ovms_version_string).group(1).strip()
            match_ov_genai_version = re.search(r"OpenVINO GenAI backend\s+([^\r\n]+)", _ovms_version_string)
            self._ov_genai_version = match_ov_genai_version.group(1).strip() if match_ov_genai_version else \
                "Not specified"
            self._ovms_build_flags = re.search(r"Bazel build flags:\s*([^\r\n]+)", _ovms_version_string).group(1).strip()

            # Get ovms container os distname
            exit_code, os_release = ovms_container.exec_run(cmd=["bash", "-c", "cat /etc/*-release"])
            os_release = os_release.decode("utf-8")
            os_distname_regex = re.compile("^PRETTY_NAME=")

            os_distname = None
            for line in os_release.splitlines():
                os_distname_match = os_distname_regex.match(line)
                if os_distname_match:
                    os_distname = line[os_distname_match.end() :].strip("\"'")
                    break
            self._os_distname = os_distname

            self._info_read = True
        except Exception as exc:
            err_msg = str(getattr(exc, "args", [""]))
            logger.error(
                "Couldn't retrieve OVMS info by running container: " f"\nException: {exc}; message: {err_msg};"
            )
            raise exc
        finally:
            ovms_container.remove(force=True)

    def _get_info_from_binary(self):
        self._os_distname = get_host_os_details()
        path_to_binary_ovms, _ = get_binaries(base_os[0], "OvmsInfo_ovms_version", tmp_dir)
        proc = Process()
        proc.disable_check_stderr()
        pre_cmd, env = get_ovms_binary_cmd_setup(base_os[0], resources_dir_path=os.path.dirname(path_to_binary_ovms))
        cmd = f"{pre_cmd}{path_to_binary_ovms} --version"
        _, stdout, _ = proc.run_and_check_return_all(cmd, env=env)
        match_ovms_version = re.search(r"Model Server\s+(\d[^\r\n]+)", stdout)
        if match_ovms_version is not None:
            self._ovms_version = match_ovms_version.group(1).strip()
        else:
            self._ovms_version = "Not specified"
        match_ov_version = re.search(r"OpenVINO backend\s+([^\r\n]+)", stdout)
        if match_ov_version is not None:
            self._ov_version = match_ov_version.group(1).strip()
        else:
            self._ov_version = "Not specified"
        match_ov_genai_version = re.search(r"OpenVINO GenAI backend\s+([^\r\n]+)", stdout)
        if match_ov_genai_version is not None:
            self._ov_genai_version = match_ov_genai_version.group(1).strip()
        else:
            self._ov_genai_version = "Not specified"
        match_ovms_build_flags = re.search(r"Bazel build flags:\s*([^\r\n]+)", stdout)
        if match_ovms_build_flags is not None:
            self._ovms_build_flags = match_ovms_build_flags.group(1).strip()
        else:
            self._ovms_build_flags = "Not specified"

    @classmethod
    def get_local_image(cls, local_image):
        image = docker.from_env().images.get(local_image)
        cls.IMAGES[local_image] = image
        return cls.IMAGES[local_image]

    @staticmethod
    def docker_pull_image_cli(image_to_pool):
        """
        Execute command line command for docker image downloading.
        This method is required to be run prior `pull_latest_image` for newly used images (unique tag).
        """
        logger.info(f"Pulling image: {image_to_pool}")
        proc = Process()
        proc.run_and_check(f"docker pull {image_to_pool}")

    @classmethod
    def pull_latest_image(cls, image_to_pull, force_pull=False):
        cls.docker_pull_image_cli(image_to_pull)  # ensure image is available on host.

        from tests.functional.utils.docker import DockerClient  # pylint: disable=import-outside-toplevel

        if image_to_pull not in cls.IMAGES or force_pull:
            repository, tag = image_to_pull.split(":")
            logger.info("Pulling image: {} tag: {}".format(repository, tag))
            image = DockerClient().pull(repository=repository, tag=tag)
            cls.IMAGES[image_to_pull] = image
        return cls.IMAGES[image_to_pull]
