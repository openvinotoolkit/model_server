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

import re
from abc import ABC, abstractmethod

import pytest

from tests.functional.utils.assertions import (
    AptInstallException,
    InstallPkgVersionException,
    OvmsTestException,
    UpgradePkgException,
)
from tests.functional.utils.logger import get_logger
from tests.functional.constants.os_type import OsType
from tests.functional.utils.process import Process
from tests.functional.constants.target_device import TargetDevice

logger = get_logger(__name__)

GPU_LIBS_TO_SKIP = ["intel-level-zero-gpu", "level-zero", "libigdgmm12"]


class PackageManager(ABC):

    def __init__(self):
        pass

    @staticmethod
    def create(base_os=OsType.Ubuntu24):
        if OsType.Redhat in base_os:
            return MicrodnfPackageManager()
        elif OsType.Ubuntu22 in base_os or OsType.Ubuntu24 in base_os:
            return AptPackageManager()

        raise NotImplementedError()

    @abstractmethod
    def get_install_cmd(self, package_name):
        raise NotImplementedError()

    def get_upgrade_pkg_cmd(self, package_name):
        raise NotImplementedError()

    @abstractmethod
    def get_install_version_cmd(self, package_name, version):
        return NotImplementedError()

    @abstractmethod
    def get_list_of_installed_packages(self, container_id):
        raise NotImplementedError()

    @abstractmethod
    def get_dependencies(self, pkg_name_list, container_id):
        raise NotImplementedError()

    def run_process(self, cmd, exception_type=OvmsTestException):
        proc = Process()
        proc.disable_check_stderr()
        stdout = proc.run_and_check(cmd, exception_type=exception_type)
        return stdout

    def get_missing_packages(self, container_pkg_list, host_pkg_list):
        missing_packages = {c: container_pkg_list[c] for c in container_pkg_list if c not in host_pkg_list}
        return missing_packages

    def get_packages_to_upgrade(self, host_pkg_list, container_pkg_list):
        packages_to_upgrade = {
            c: container_pkg_list[c]
            for c in container_pkg_list
            if c in host_pkg_list and host_pkg_list[c]["version"] < container_pkg_list[c]["version"]
        }
        return packages_to_upgrade

    def install_missing_packages_on_host(self, container_pkg_list, host_pkg_list, missing_pkg_list):
        pkgs_to_install = {}
        for pkg, version in missing_pkg_list.items():
            if self.context.target_device == TargetDevice.GPU and pkg in GPU_LIBS_TO_SKIP:
                logger.debug(f"Skip package {pkg} - will be downloaded in GpuDriverInstaller.")
                continue
            logger.debug(f"Installing {pkg} ...")
            pkgs_to_install.update({pkg: version})

        # Map each element to string: <pkg_name>=<version>:
        pkgs_to_install = list(map(lambda x: f"{x[0]}={x[1]['version']}", pkgs_to_install.items()))

        # It is important to install packages all at once with explicit version:
        # it would help to minimalize risk of version mismatch of some low level packages
        if len(pkgs_to_install) != 0:
            cmd = self.install_cmd.format(" ".join(pkgs_to_install))  # install packages all at once
            _ = self.run_process(cmd, exception_type=AptInstallException)
            after_install_host_pkg_list = self.get_list_of_installed_packages(container_id=None)
            assert not self.get_missing_packages(container_pkg_list, after_install_host_pkg_list)
            return host_pkg_list
        else:  # Return if there are no more pkgs_to_install except those defined in GPU_LIBS_TO_SKIP
            return host_pkg_list

    def upgrade_packages(self, packages_to_upgrade, container_pkg_list):
        for key, value in packages_to_upgrade.items():
            logger.debug(f"Upgrading pkg '{key}' to version '{value['version']}' ...")
            cmd = self.get_install_version_cmd(key, value["version"])
            try:
                self.run_process(cmd, exception_type=InstallPkgVersionException)
            except InstallPkgVersionException as e:
                cmd, retcode, stdout, stderr = e.get_process_details()
                if "The following packages have unmet dependencies" in stdout:
                    logger.debug(f"Upgrading all system packages ...")
                    self.run_process(self.upgrade_cmd, exception_type=UpgradePkgException)
                    host_packages = self.get_list_of_installed_packages(container_id=None)
                    if not self.get_packages_to_upgrade(host_packages, container_pkg_list):
                        break
                elif f"Version '{value['version']}' for '{key}' was not found" in stderr:
                    logger.debug(f"Upgrading package {key}")
                    upgrade_pkg_cmd = self.get_upgrade_pkg_cmd(key)
                    self.run_process(upgrade_pkg_cmd, exception_type=UpgradePkgException)
                else:
                    raise InstallPkgVersionException(f"Found exception in executing cmd: {cmd}; stdout: {stdout}")
        host_packages = self.get_list_of_installed_packages(container_id=None)
        pkgs = self.get_packages_to_upgrade(host_packages, container_pkg_list)
        if not pkgs:
            return
        else:
            for key, value in pkgs.items():
                logger.warning(
                    f"Failed to upgrade package {key}. Continue with the current version: {value['version']}"
                )


class DnfPackageManager(PackageManager):

    def __init__(self):
        self.update_cmd = "dnf update -y"
        self.install_cmd = "dnf install -y"
        self.upgrade_cmd = "dnf upgrade -y"

    def get_install_cmd(self, package_name):
        return f"{self.install_cmd} {package_name}"

    def get_install_version_cmd(self, package_name, version):
        return f"{self.install_cmd} {package_name}-{version}"

    def get_upgrade_pkg_cmd(self, package_name):
        return f"{self.upgrade_cmd} {package_name}"

    def get_list_of_installed_packages(self, container_id):
        process = Process()
        error_code, stdout, stderr = process.run(f"docker exec {container_id} rpm --query --all")
        assert error_code == 0, f"Detected unexpected return code: {error_code} (stderr: {stderr})"

        package_list = stdout.splitlines()
        detected_packages = {}

        # general regex e.g. publicsuffix-list-dafsa-20180723-1.el8.noarch
        # also handles dots in package name e.g. python3.12-pip-wheel-23.2.1-5.el9.noarch
        rpm_list_pkg_regexp = re.compile(r"^([\w\.\-\+]+)\-([\w\.]+)\-([\w\.\+]+)\.(\w+)$")
        # regex for packages that have no architecture info in name e.g. gpg-pubkey-fd431d51-4ae0493b
        rpm_list_pkg_no_arch_regexp = re.compile(r"^([\w\.\-]+)\-(\w+)\-(\w+)$")

        for pkg in package_list:
            match = rpm_list_pkg_regexp.match(pkg)
            if match:
                name, version, release, arch = match.groups()
            else:
                match = rpm_list_pkg_no_arch_regexp.match(pkg)
                assert match, f"Unable to parse package info: {pkg}"
                name, version, release = match.groups()
                arch = "noarch"
            detected_packages[name] = {"arch": arch, "version": version}

        return detected_packages

    def get_dependencies(self, pkg_name_list, container_id):
        result = set()
        process = Process()
        cmd = f"docker exec {container_id} dnf"
        error_code, stdout, stderr = process.run(cmd)
        assert error_code == 0, (
            f"Please install dnf package. " f"Detected unexpected return code: {error_code} (stderr: {stderr})"
        )
        # Example of output for curl tool:
        #   glibc-0:2.28-225.el8.i686
        #   glibc-0:2.28-225.el8.x86_64
        #   libcurl-0:7.61.1-30.el8_8.2.x86_64
        #   openssl-libs-1:1.1.1k-9.el8_7.x86_64
        #   zlib-0:1.2.11-21.el8_7.x86_64
        dnf_repoquery_regexp = re.compile(r"^([\w\-\+]+)\-\d\:([\w\.]+)\-([\w\.\+]+)\.(\w+)$")
        for pkg_name in pkg_name_list:
            cmd = f"docker exec {container_id} dnf repoquery --requires --resolve {pkg_name}"
            error_code, stdout, stderr = process.run(cmd)
            assert error_code == 0, f"Detected unexpected return code: {error_code} (stderr: {stderr})"
            for line in stdout.splitlines():
                match = dnf_repoquery_regexp.match(line)
                assert match, f"Unable to parse package info: {line}"
                name = match.group(1)
                result.add(name)
        return list(result)


class MicrodnfPackageManager(DnfPackageManager):

    def __init__(self):
        self.update_cmd = "microdnf update -y"
        self.install_cmd = "microdnf install -y"
        self.upgrade_cmd = "microdnf upgrade -y"


class YumPackageManager(PackageManager):

    def __init__(self):
        self.update_cmd = "sudo yum update -y"
        self.install_cmd = "sudo yum install {} -y"
        self.upgrade_cmd = "sudo yum upgrade -y"

    def get_install_cmd(self, package_name):
        return f"yum clean all; yum install {package_name} -y"

    def get_install_version_cmd(self, package_name, version):
        return f"sudo yum install {package_name}-{version} -y"

    def get_upgrade_pkg_cmd(self, package_name):
        return f"sudo yum upgrade {package_name} -y"

    def get_list_of_installed_packages(self, container_id):
        process = Process()
        error_code, stdout, stderr = process.run(f"docker exec {container_id} yum list installed | grep installed")
        assert error_code == 0, f"Detect unexpected return code: {error_code} (stderr: {stderr})"

        package_list = stdout.splitlines()
        detected_packages = {}
        for pkg in package_list:
            pkg_name, pkg_version, _ = pkg.split()
            name, arch = pkg_name.split(".")
            detected_packages[name] = {"arch": arch, "version": pkg_version}

        return detected_packages

    def get_dependencies(self, pkg_name_list, container_id):
        for pkg_name in pkg_name_list:
            if pkg_name != "curl":
                pytest.skip(reason=f"Unable to collect dependencies for {pkg_name}!")
        return []


class AptPackageManager(PackageManager):

    def __init__(self):
        self.update_cmd = "sudo apt update -y"
        self.install_cmd = "sudo apt install {} -y"
        self.upgrade_cmd = "sudo apt upgrade -y"

    def get_install_cmd(self, package_name):
        return f"apt-get update -y && apt-get install -y {package_name}"

    def get_install_version_cmd(self, package_name, version):
        return f"sudo apt install {package_name}={version} -y"

    def get_upgrade_pkg_cmd(self, package_name):
        return f"sudo apt upgrade {package_name} -y"

    def get_list_of_installed_packages(self, container_id):
        list_installed_cmd = "apt list --installed | grep installed"
        if container_id is None:  # get installed packages on host
            cmd = list_installed_cmd
        else:  # get installed packages in container
            cmd = f"docker exec {container_id} {list_installed_cmd}"
        apt_list_pkg_regexp = re.compile(r"^([^\/]+)\/[^\s]+\s([^\s]+)\s(\w+)")
        process = Process()
        error_code, stdout, stderr = process.run(cmd)
        assert error_code == 0, f"Detect unexpected return code: {error_code} (stderr: {stderr})"

        package_list = stdout.splitlines()
        detected_packages = {}
        for pkg in package_list:
            # ppp/focal-updates,focal-security,now 2.4.7-2+4.1ubuntu5.1 amd64 [installed,automatic]
            # pptp-linux/focal,now 1.10.0-1build1 amd64 [installed,automatic]
            # procps/focal,now 2:3.3.16-1ubuntu2 amd64 [installed,upgradable to: 2:3.3.16-1ubuntu2.2]
            # psmisc/focal,now 23.3-1 amd64 [installed,automatic]
            # libgnutls30/now 3.6.13-2ubuntu1.3 amd64 [installed,upgradable to: 3.6.13-2ubuntu1.6]
            # login/now 1:4.8.1-1ubuntu5.20.04 amd64 [installed,upgradable to: 1:4.8.1-1ubuntu5.20.04.1]
            # passwd/now 1:4.8.1-1ubuntu5.20.04 amd64 [installed,upgradable to: 1:4.8.1-1ubuntu5.20.04.1]
            match = apt_list_pkg_regexp.search(pkg)
            assert match, f"Unable to parse package info: {pkg}"
            name, version, arch = match.groups()
            detected_packages[name] = {"arch": arch, "version": version}

        return detected_packages

    def get_dependencies(self, pkg_name_list, container_id):
        result = []
        process = Process()
        # Example of output for curl tool:
        #    WARNING: apt does not have a stable CLI interface. Use with caution in scripts.
        #
        #    Reading package lists...
        #    Building dependency tree...
        #    Reading state information...
        #    The following packages were automatically installed and are no longer required:
        #      (...)
        #      libsasl2-modules-db libsqlite3-0 libssh-4 libwind0-heimdal publicsuffix
        #    Use 'apt autoremove' to remove them.
        #    The following packages will be REMOVED:
        #      curl
        for pkg_name in pkg_name_list:
            error_code, stdout, stderr = process.run(f"docker exec {container_id} apt -s remove {pkg_name}")
            assert error_code == 0, f"Detect unexpected return code: {error_code} (stderr: {stderr})"
            header = "The following packages were automatically installed and are no longer required:"
            footer = "Use 'apt autoremove' to remove them."
            start = stdout.find(header) + len(header)
            end = stdout.find(footer)
            lib_list_str = stdout[start:end]
            for line in lib_list_str.splitlines():
                libs = line.strip().split()
                for lib in libs:
                    if lib not in result:
                        result.append(lib)
        return result
