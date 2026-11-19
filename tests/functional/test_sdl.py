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

import docker
import pytest

from tests.functional.config import container_proxy, is_nginx_mtls
from tests.functional.constants.components import OvmsComponents
from tests.functional.constants.os_type import OsType, UBUNTU
from tests.functional.constants.ovms import Ovms
from tests.functional.constants.ovms_binaries import get_binaries
from tests.functional.constants.ovms_type import OvmsType
from tests.functional.constants.requirements import Requirements
from tests.functional.constants.target_device_configuration import nginx_mtls_not_supported_for_test
from tests.functional.fixtures.server import start_ovms
from tests.functional.models.models_classic import Resnet50Int8OvHf
from tests.functional.object_model.ovms_info import OvmsInfo
from tests.functional.object_model.ovms_params import OvmsParams
from tests.functional.object_model.package_manager import PackageManager
from tests.functional.object_model.test_environment import TestEnvironment
from tests.functional.utils.assertions import SdlException
from tests.functional.utils.context import Context
from tests.functional.utils.docker import DockerContainer
from tests.functional.utils.logger import get_logger, step
from tests.functional.utils.sdl_whitelists_helpers import get_whitelist, validate_whitelists

logger = get_logger(__name__)


@pytest.mark.priority_high
@pytest.mark.components(OvmsComponents.OVMS)
@pytest.mark.reqids(Requirements.sdl)
class TestSdl:

    @staticmethod
    def get_directory_libraries_content(ovms, directory):
        ls_output = ovms.execute_and_check(f"ls {directory} -l")
        lib_folder_content = ls_output.splitlines()
        libraries = [x for x in lib_folder_content if "lib" in x]
        return libraries


    @staticmethod
    def generate_libraries_duplication_report(ovms, libraries, ovms_lib_path):
        duplication_report = {}
        library_name_regexp = re.compile(r"(lib.+?)\.so")
        library_full_name_regexp = re.compile(r"(lib.+)")
        library_size_regexp = re.compile(r"[\w-]{10} \d \w+ \w+\s+(\d+)\s")
        for library in libraries:
            library_name = library_name_regexp.search(library).group(1)

            duplicated_libraries = [
                library_full_name_regexp.search(x).group(1)
                for x in libraries
                if library_name_regexp.search(x).group(1) == library_name
            ]

            if library_name not in duplication_report and len(duplicated_libraries) > 1:
                library_info = []
                for file in duplicated_libraries:
                    _, output = ovms.execute_command(f"file -b {ovms_lib_path}/{file}")
                    size = library_size_regexp.search(library).group(1)
                    library_info.append((file, size, output))

                tmp_library_info = [(file, size) for file, size, info in library_info if "ASCII text" not in info]
                if len(tmp_library_info) > 1:
                    duplication_report[library_name] = tmp_library_info
        return duplication_report

    @staticmethod
    def start_base_image_docker(context: Context):
        client = docker.APIClient()
        docker_inspect_output = client.inspect_image(context.ovms_image)
        base_image = docker_inspect_output["Config"]["Labels"]["base-image"]

        OvmsInfo.docker_pull_image_cli(base_image)
        base_image_container = DockerContainer(
            container=DockerContainer(None).client.run(base_image, "sleep inf", detach=True)
        )
        context.test_objects.append(base_image_container.cleanup)
        return base_image_container.id

    @staticmethod
    def collect_ldd_dependencies(ovms, binary_name, cwd=None):
        stdout = ovms.execute_and_check(f"ldd {binary_name}", cwd=cwd)
        logger.info(stdout)
        dependency_list = [
            m.groups() for line in stdout.splitlines() for m in [re.search(r"(.+) => (.+) \(.+\)", line.strip())] if m
        ]
        return dependency_list

    @staticmethod
    def find_library_from_binary_name(binary_name, libraries_list) -> tuple:
        for library_name, library_full_path in libraries_list:
            if binary_name in library_name:
                return library_name, library_full_path
        raise SdlException(f"Did not found {binary_name} in the {libraries_list}")

    def collect_additional_ldd_dependencies(self, ovms, additional_tools, dependency_list):
        additional_dependency_list = []
        for additional_tool in additional_tools:
            library_name, library_full_path = self.find_library_from_binary_name(additional_tool, dependency_list)
            additional_dependency_list.append((library_name, library_full_path))
            additional_dependency_list += self.collect_ldd_dependencies(ovms, library_full_path)
        return additional_dependency_list

    @pytest.mark.api_on_commit
    @pytest.mark.ovms_types_supported_for_test(OvmsType.DOCKER)
    def test_validate_libraries_duplicates(self, context: Context):
        """
        <b>Description:</b>
        Check if docker image do not use duplicated library files

        <b>Expected results:</b>
        Docker use single version of each library

        <b>Steps:</b>
        1. List all libraries found in /ovms/lib folder
        3. Verify that content of /ovms/lib contains single file of each library
        """
        step("List all libraries found in /ovms/lib folder")
        envs = {"http_proxy": container_proxy, "https_proxy": container_proxy}
        result = start_ovms(context, OvmsParams(models=[Resnet50Int8OvHf()]), environment=envs)
        _, ovms_lib_binary_path = Ovms.get_ovms_binary_paths(context.ovms_type)
        libraries = self.get_directory_libraries_content(result.ovms, ovms_lib_binary_path)
        libraries = [x for x in libraries if "->" not in x]

        step("Verify that content of /ovms/lib contains single file of each library")
        duplication_report = self.generate_libraries_duplication_report(result.ovms, libraries, ovms_lib_binary_path)
        if duplication_report:
            raise SdlException(f"Detect duplicated objects. Report: {duplication_report}")

    @pytest.mark.api_on_commit
    @pytest.mark.ovms_types_supported_for_test(OvmsType.DOCKER)
    @pytest.mark.os_types_supported_for_test(OsType.Ubuntu24, OsType.Redhat)
    @nginx_mtls_not_supported_for_test()
    def test_unnecessary_packages_in_container(self, context: Context):
        """
        <b>Description:</b>
        Bloating containers with unnecessary software could possibly increase the attack surface of the container.
        This also voids the concept of minimal and slim down versions of container images.

        <b>Input data:</b>
        1. OVMS image

        <b>Expected results:</b>
        Test passes when there are no unnecessary packages installed in container.

        <b>Steps:</b>
        1. List all packages installed in the OVMS docker container
        2. List all packages installed in the base image docker container and remove duplicates from original list
        3. List all packages installed for additional tools and remove duplicates from original list
        4. Compare detected packages against whitelist
        """
        packet_manager = PackageManager.create(context.base_os)

        step("List all packages installed in the OVMS docker container")
        result = start_ovms(context, OvmsParams(models=[Resnet50Int8OvHf()]))
        detected_packages = packet_manager.get_list_of_installed_packages(result.ovms.get_short_id())

        step("List all packages installed in the base image docker container and remove duplicates from original list")
        base_image_container_id = self.start_base_image_docker(context)
        detected_packages_base_image = packet_manager.get_list_of_installed_packages(base_image_container_id)
        for package_name in detected_packages_base_image:
            if package_name in detected_packages:
                detected_packages.pop(package_name)

        step("List all packages installed for additional tools and remove duplicates from original list")
        additional_tools = []
        if UBUNTU in context.base_os:
            additional_tools.append("curl")
        if is_nginx_mtls:
            additional_tools.append("nginx")
        if OsType.Redhat in context.base_os:
            # dnf tool required to check indirect dependencies
            exit_code, _ = result.ovms.execute_command("dnf --version")
            if exit_code != 0:
                result.ovms.install_package(base_os=context.base_os, tool_name="dnf")
        indirect_dependency_packages_list = packet_manager.get_dependencies(
            additional_tools, result.ovms.get_short_id()
        )

        for indirect_dependency_package in indirect_dependency_packages_list:
            if indirect_dependency_package in detected_packages:
                detected_packages.pop(indirect_dependency_package)

        step("Compare detected packages against whitelist")
        validate_whitelists(context, whitelist_name="whitelisted_packages", detected_objects=detected_packages)

    @pytest.mark.api_on_commit
    @pytest.mark.ovms_types_supported_for_test(OvmsType.DOCKER)
    def test_validate_whitelisted_libraries(self, context: Context):
        """
        <b>Description:</b>
        Check if docker image uses only whitelisted libraries

        <b>Expected results:</b>
        Docker do not use additional libraries

        <b>Steps:</b>
        1. List all libraries found in /ovms/lib folder
        2. Compare detected libraries against whitelist
        """
        step("List all libraries found in /ovms/lib folder")
        result = start_ovms(context, OvmsParams(models=[Resnet50Int8OvHf()]))
        _, ovms_lib_binary_path = Ovms.get_ovms_binary_paths(context.ovms_type)
        libraries = self.get_directory_libraries_content(result.ovms, ovms_lib_binary_path)

        detected_libraries = []
        unexpected_objects = []
        library_name_regexp = re.compile(r"(lib.+?\.so)")
        for library in libraries:
            match_library = library_name_regexp.search(library)
            if match_library is not None:
                detected_libraries.append(match_library.group(1))
            else:
                unexpected_objects.append(library)
        if unexpected_objects:
            raise SdlException(f"Detect unexpected object in /ovms/lib folder: {unexpected_objects}")

        step("Verify detected packages against whitelist")
        validate_whitelists(context, whitelist_name="whitelisted_libraries", detected_objects=detected_libraries)

    @pytest.mark.api_on_commit
    @pytest.mark.ovms_types_supported_for_test(OvmsType.DOCKER)
    def test_validate_dynamic_libraries_list(self, context: Context):
        """
        <b>Description:</b>
        Make sure that OVMS does not use prohibited 3rd party components or dependencies - check list of used dynamic
        libraries using ldd command

        <b>Input data:</b>
        Running OVMS

        <b>Expected results:</b>
        There is no unexpected components or dependencies

        <b>Steps:</b>
        1. Collect list of libraries used by OVMS with ldd command
        2. Collect list of libraries used by additional tools with ldd command
        3. Compare detected libraries against whitelists
        """
        step("Collect list of used libraries with ldd command")
        result = start_ovms(context, OvmsParams(models=[Resnet50Int8OvHf()]))
        ovms_binary_path, ovms_lib_binary_path = Ovms.get_ovms_binary_paths(context.ovms_type)
        ovms_work_path = result.ovms.container_folder if context.ovms_type == OvmsType.BINARY else None
        ovms_dependency_list = self.collect_ldd_dependencies(result.ovms, ovms_binary_path, cwd=ovms_work_path)

        step("Collect list of libraries used by additional tools with ldd command")
        additional_tools = ["curl"]
        additional_dependency_list = self.collect_additional_ldd_dependencies(
            result.ovms, additional_tools, ovms_dependency_list
        )

        step("Compare detected libraries against whitelists")
        detected_ovms_dynamic_libraries = []
        detected_dynamic_libraries = []
        library_name_regexp = re.compile(r"(lib.+?\.so)")
        for library_info in ovms_dependency_list:
            if library_info not in additional_dependency_list:
                library_name = library_name_regexp.search(library_info[0]).group(1)
                library_path = library_info[1]
                if ovms_lib_binary_path in os.path.abspath(library_path):
                    detected_ovms_dynamic_libraries.append(library_name)
                else:
                    detected_dynamic_libraries.append(library_name)

        validate_whitelists(
            context,
            whitelist_name="whitelisted_libraries",
            detected_objects=detected_ovms_dynamic_libraries,
            check_missing_objects=False,
        )
        validate_whitelists(
            context, whitelist_name="whitelisted_dynamic_libraries", detected_objects=detected_dynamic_libraries
        )

        whitelisted_objects = get_whitelist(context, whitelist_name="whitelisted_libraries")
        duplicated_libraries = set(x for x in detected_dynamic_libraries if x in whitelisted_objects)
        if duplicated_libraries:
            raise SdlException(
                f"Detected unexpected duplicates in {ovms_lib_binary_path} folder: " f"{duplicated_libraries}"
            )

    @pytest.mark.api_on_commit
    @pytest.mark.ovms_types_supported_for_test(OvmsType.BINARY)
    @pytest.mark.os_types_supported_for_test(OsType.Windows)
    def test_validate_whitelisted_libraries_windows(self, context: Context):
        """
        <b>Description:</b>
        Check if windows binary package uses only whitelisted libraries

        <b>Expected results:</b>
        Windows binary uses only whitelisted libraries

        <b>Steps:</b>
        1. List all libraries found in ovms binary path
        2. Compare detected libraries against a whitelist
        """
        step("List all libraries found in ovms binary path")
        ovms_binary_path, _ = get_binaries(
            context.base_os,
            context.test_object_name,
            TestEnvironment.current.base_dir,
        )

        dll_files = [
            file for _, _, files in os.walk(os.path.dirname(ovms_binary_path))
            for file in files if file.endswith((".dll", ".exe")) and file != "ovms.exe"
        ]

        step("Verify detected packages against whitelist")
        validate_whitelists(context, whitelist_name="whitelisted_libraries", detected_objects=dll_files)
