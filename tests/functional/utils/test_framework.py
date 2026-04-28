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
import pytest
import re
import shutil
import traceback

from tests.functional.utils.assertions import CreateVenvError, PipInstallError
from tests.functional.utils.git_operations import clone_git_repository
from tests.functional.utils.logger import get_logger
from tests.functional.utils.process import Process, WindowsProcess
from tests.functional.utils.helpers import get_xdist_worker_count, generate_test_object_name

from tests.functional.config import (
    build_test_image,
    language_models_enabled,
    machine_is_reserved_for_test_session,
    mediapipe_disable,
    python_disable,
    run_ovms_with_opencl_trace,
    run_ovms_with_valgrind,
    windows_python_version,
)

logger = get_logger(__name__)

STACK_TRACE_SEARCH_PATTERN = "site-packages"


class FrameworkMessages:
    NOT_IMPLEMENTED = "NOT IMPLEMENTED"
    NEXT_VERSION = "NEXT VERSION"
    NOT_TO_BE_REPORTED_IF_SKIPPED = "NOT TO BE REPORTED IF SKIPPED"
    TWO_USERS_CANT_MOUNT_ON_WINDOWS = "Cannot create mount point by two nctl users on one Windows user"
    TWO_USERS_CANT_MOUNT_ON_NON_LINUX = "Cannot create mount point by two nctl users on one non-Linux user"
    TEST_ISSUE = "TEST ISSUE"
    FEATURE_NOT_READY = "FEATURE NOT READY"
    NOT_SUPPORTED_IN_SAFARI = "Not possible to automate on Safari browser"
    NOT_EXECUTED = "NOT EXECUTED"
    OVMS_C_REPO_ABSENT = "OVMS-C REPO ABSENT"
    OVMS_O_REPO_ABSENT = "OVMS-O REPO ABSENT"
    FUZZING_TESTS_NOT_ENABLED = "FUZZING TESTS NOT ENABLED"
    UAT_TESTS_NOT_ENABLED = "UAT TESTS NOT ENABLED"
    OV_TESTS_NOT_ENABLED = "OV TESTS NOT ENABLED"
    OVMS_MEDIA_TESTS_NOT_ENABLED = "OVMS MEDIAPIPE TESTS NOT ENABLED"
    OS_NOT_SUPPORTED = "{} OS NOT SUPPORTED"
    MACHINE_NOT_RESERVED_FOR_TEST_SESSION = "MACHINE NOT RESERVED FOR TEST SESSION"
    RUN_IN_PARALLEL= "RUN IN PARALLEL"
    TEST_IMAGE_NOT_BUILD = "TEST IMAGE NOT BUILD"
    LANGUAGE_MODELS_DISABLED = "LANGUAGE MODELS DISABLED"
    MEDIAPIPE_DISABLED = "MEDIAPIPE DISABLED"
    PYTHON_DISABLED = "PYTHON DISABLED"
    OPENSHIFT_SERVICE_MESH_ENABLED = "OPENSHIFT SERVICE MESH ENABLED"
    OPENSHIFT_SERVICE_MESH_DISABLED = "OPENSHIFT SERVICE MESH DISABLED"
    ADD_NOTEBOOK_K8S_DISABLED = "ADD NOTEBOOK K8S DISABLED"
    BUILD_AND_VERIFY_PACKAGE_DISABLED = "BUILD_AND_VERIFY_PACKAGE_DISABLED"
    VLLM_TESTS_NOT_ENABLED = "vLLM TESTS NOT ENABLED"
    NGINX_IMAGE_NOT_SUPPORTED = "NGINX IMAGE NOT SUPPORTED"
    KFS_GET_MODEL_STATUS_NOT_SUPPORTED = "GetModelStatus is not supported in KFS"
    LM_ACCURACY_TESTS_NOT_ENABLED = "Language models accuracy tests not enabled"
    CLILOADER_DISABLED = "Cliloader disabled"
    VALGRIND_DISABLED = "Valgrind disabled"


class TestStatus:
    RESULT_PASS = "PASS"
    RESULT_FAIL = "FAIL"
    RESULT_SKIPPED = "SKIPPED"
    RESULT_NOT_EXECUTED = "NOT_EXECUTED"
    RESULT_NOT_IMPLEMENTED = FrameworkMessages.NOT_IMPLEMENTED.replace(" ", "_")
    RESULT_NEXT_VERSION = FrameworkMessages.NEXT_VERSION.replace(" ", "_")
    RESULT_UNKNOWN = "UNKNOWN"
    RESULT_NOT_TO_BE_REPORTED = FrameworkMessages.NOT_TO_BE_REPORTED_IF_SKIPPED.replace(" ", "_")
    RESULT_TEST_ISSUE = FrameworkMessages.TEST_ISSUE.replace(" ", "_")
    RESULT_FEATURE_NOT_READY = FrameworkMessages.FEATURE_NOT_READY.replace(" ", "_")


def get_msg_with_stack_trace(msg):
    # get stack trace for ovms framework functions only
    current_stack_trace = []
    for trace_line in traceback.format_stack()[::-1]:
        if STACK_TRACE_SEARCH_PATTERN in trace_line:
            break
        current_stack_trace.append(trace_line)
    return f"{msg}\nStackTrace:\n{' '.join(current_stack_trace[::-1])}"


def skip_if_runtime(condition, msg=FrameworkMessages.NOT_TO_BE_REPORTED_IF_SKIPPED):
    if condition:
        pytest.skip(reason=get_msg_with_stack_trace(msg))


def skip_if(condition, msg=FrameworkMessages.NOT_TO_BE_REPORTED_IF_SKIPPED):
    return pytest.mark.skipif(condition, reason=get_msg_with_stack_trace(msg))


def skip_not_implemented():
    return pytest.mark.skip(reason=get_msg_with_stack_trace(FrameworkMessages.NOT_IMPLEMENTED))


def skip_if_language_models_not_enabled():
    return skip_if(not language_models_enabled, msg=FrameworkMessages.LANGUAGE_MODELS_DISABLED)


def skip_if_mediapipe_disabled():
    return skip_if(mediapipe_disable, msg=FrameworkMessages.MEDIAPIPE_DISABLED)


def skip_if_python_disabled():
    return skip_if(python_disable, msg=FrameworkMessages.PYTHON_DISABLED)


def skip_if_cliloader_disabled():
    return skip_if(not run_ovms_with_opencl_trace, msg=FrameworkMessages.CLILOADER_DISABLED)


def skip_if_valgrind_disabled():
    return skip_if(not run_ovms_with_valgrind, msg=FrameworkMessages.VALGRIND_DISABLED)


def skip_if_build_test_image_not_enabled():
    return skip_if(not build_test_image, msg=FrameworkMessages.TEST_IMAGE_NOT_BUILD)


def skip_if_machine_is_not_reserved_for_test_session():
    return skip_if(not machine_is_reserved_for_test_session,
                   msg=FrameworkMessages.MACHINE_NOT_RESERVED_FOR_TEST_SESSION)


def skip_if_run_in_parallel():
    return skip_if(not is_single_threaded(), msg=FrameworkMessages.RUN_IN_PARALLEL)


def get_xdist_worker_string():
    return os.environ.get("PYTEST_XDIST_WORKER", "master")


def is_xdist_master():
    return get_xdist_worker_string() == "master"


def current_pytest_test_case(separator="_"):
    test_case = os.environ.get("PYTEST_CURRENT_TEST")
    test_case = test_case.split("::")[-1]
    test_case = re.sub('[^a-z0-9]+', separator, test_case)
    return test_case


def get_test_object_prefix():
    # recover e.g.: tmp_k8sworker_342_ from test_object_name
    object_name_match = re.match(r"([^\W_]+.[^\W_]+.[^\W_]+.)[^\W_]+.[^\W_]+", generate_test_object_name())
    assert object_name_match, "Can't' find prefix match for test_object_name"
    test_object_prefix = object_name_match.group(1)
    return test_object_prefix


def is_single_threaded():
    return get_xdist_worker_count() <= 1


def get_parameter_from_item(item, param_name, default=None):
    _callspec = getattr(item, 'callspec', None)
    return _callspec.params.get(param_name, default) if _callspec else default


def create_venv_and_install_packages(
        work_dir,
        requirements_string=None,
        requirements_file_path=None,
        venv_dir_name=".venv",
        pip_additional_options="",
        **kwargs
):
    print(f"Creating virtualenv in path: {work_dir}")
    venv_dir = os.path.join(work_dir, venv_dir_name)
    os.makedirs(venv_dir, exist_ok=True)

    process = Process()
    process.disable_check_stderr()

    python_version = kwargs.get("python_version", "3")
    if isinstance(process, WindowsProcess):
        activate_path = os.path.join(venv_dir, "Scripts", "activate.bat")
        create_venv_cmd = f"virtualenv {venv_dir} --python=python{windows_python_version}"
        install_pip_cmd = f"{activate_path} && python -m pip install -U pip"
    else:
        activate_path = os.path.join(venv_dir, "bin", "activate")
        create_venv_cmd = f"python{python_version} -m venv {venv_dir}"
        install_pip_cmd = f". {activate_path} && pip3 install -U pip"

    process.run_and_check(create_venv_cmd, exception_type=CreateVenvError)
    process.run_and_check(install_pip_cmd, exception_type=PipInstallError, timeout=900)
    if requirements_string is not None or requirements_file_path is not None:
        if requirements_file_path is not None:
            if isinstance(process, WindowsProcess):
                requirements_string = (f"pip install -r {requirements_file_path} --trusted-host download.pytorch.org "
                                       f"--trusted-host storage.openvinotoolkit.org")
            else:
                requirements_string = f"pip install -r {requirements_file_path} {pip_additional_options}"

        install_requirements_cmd = f"{activate_path} && {requirements_string}"\
            if isinstance(process, WindowsProcess) else \
            f". {activate_path} && {requirements_string}"
        print(f"Installing requirements with command: {install_requirements_cmd}\n")
        _, stdout, stderr = process.run_and_check_return_all(
            install_requirements_cmd,
            cwd=work_dir,
            exception_type=PipInstallError,
        )
        print(f"Stdout: {stdout}\nStderr {stderr}")

    return activate_path


def create_venv_and_install_packages_from_git_repo(
        repo_url,
        repo_path,
        repo_branch,
        requirements_string=None,
        requirements_file_path=None,
        venv_dir_name=".venv",
        pip_additional_options="",
        commit_sha=None,
        **kwargs
):
    print(f"Clone repository: {repo_url}")
    clone_git_repository(repo_url=repo_url, repo_path=repo_path, repo_branch=repo_branch, commit_sha=commit_sha)

    create_venv_and_install_packages(
        repo_path,
        requirements_string,
        requirements_file_path,
        venv_dir_name,
        pip_additional_options,
        **kwargs
    )


def change_dir_permissions(dir_path, permissions=0o777):
    for root, dirs, files in os.walk(dir_path):
        for d in dirs:
            os.chmod(os.path.join(root, d), permissions)
        for f in files:
            os.chmod(os.path.join(root, f), permissions)


def remove_dir_contents(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
