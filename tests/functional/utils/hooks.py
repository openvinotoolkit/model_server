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

import itertools
import os
import re
import shutil
import sys
import time
import warnings
import pytest

from collections import Counter, defaultdict, namedtuple
from docker import errors as docker_errors
from itertools import groupby
from pathlib import Path
from _pytest.mark import Mark, MarkDecorator
from _pytest.python import Function

from tests.functional import config
from tests.functional.models.models_library import ModelsLib, ModelsLibrary
from tests.functional.utils.download import wget_file
from tests.functional.utils.reservation_manager.args import parse_args
from tests.functional.utils.reservation_manager.manager import Manager as ReservationManager
from tests.functional.config import (
    build_test_image,
    c_api_wrapper_dir,
    cleanup_env_on_startup,
    components_ids,
    exclude_components_ids,
    exclude_req_ids,
    force_generate_new_ssl_certs,
    global_tmp_dir_default,
    http_proxy,
    https_proxy,
    is_nginx_mtls,
    no_proxy,
    ovms_c_repo_path,
    ovms_file_locks_dir,
    performance_test_timeout_minutes,
    machine_is_reserved_for_test_session,
    req_ids,
    run_ovms_with_opencl_trace,
    run_ovms_with_valgrind,
    target_devices,
    tests_priority_list,
    tmp_dir,
)
from tests.functional.constants.os_type import get_host_os, OsType, UBUNTU
from tests.functional.constants.os_version import os_type_to_base_image_binary_docker
from tests.functional.constants.ovms import (
    BASE_OS_PARAM_NAME,
    CURRENT_TARGET_DEVICE_DICT_ARGUMENT,
    OVMS_TYPE_PARAM_NAME,
    TARGET_DEVICE_PARAM_NAME,
    TMP_REPOS_DIR_ARGUMENT,
    USES_MAPPING_PARAM_NAME,
)
from tests.functional.constants.ovms_images import (
    calculate_ovms_binary_image_name,
    calculate_ovms_capi_image_name,
    calculate_ovms_test_image_name,
    calculate_ovms_image_name,
    get_ovms_calculated_images,
    GPU_INSTALL_DRIVER_VERSION,
    GPU_INSTALL_SCRIPTS,
)
from tests.functional.constants.ovms_type import (
    OvmsType,
    OVMS_BINARY_DEPENDENCIES,
    OVMS_BINARY_PACKAGE_EXTENSIONS,
    OVMS_BINARY_PACKAGE_NAME,
    OVMS_CAPI_DEPENDENCIES,
    OVMS_CAPI_TOOLS_DEPENDENCIES,
)
from tests.functional.constants.paths import Paths
from tests.functional.constants.target_device import MAX_WORKERS_PER_TARGET_DEVICE, TargetDevice
from tests.functional.constants.ovms_binaries import calculate_ovms_binary_name
from tests.functional.object_model.ovms_info import OvmsInfo
from tests.functional.utils.core import TmpDir
from tests.functional.utils.docker import DockerClient, DockerContainer, DOCKER_CONTAINER_TMP_PATH
from tests.functional.utils.environment_info import EnvironmentInfo
from tests.functional.utils.logger import get_logger
from tests.functional.utils.marks import (
    MarkConditionalRunType,
    MarkGeneral,
    MarkPriority,
    MarkRunType,
    MarkTestParameters,
)
from tests.functional.utils.ov_hf_downloader import OVHfDownloader
from tests.functional.utils.process import PID_STATE_ZOMBIE, Process, get_pid_name, get_pid_status
from tests.functional.utils.test_framework import change_dir_permissions, get_test_object_prefix, is_xdist_master
from tests.functional.utils.helpers import get_base_device
from tests.functional.object_model.ovsa import OvsaCerts

logger = get_logger(__name__)


timeout_dict = defaultdict(
    lambda: 5 * 60,
    {
        MarkRunType.TEST_MARK_ON_COMMIT: 3 * 60,
        MarkRunType.TEST_MARK_REGRESSION: 5 * 60,
        MarkRunType.TEST_MARK_REGRESSION_SINGLE: 5 * 60,
        MarkRunType.TEST_MARK_REGRESSION_WEEKLY: 5 * 60,
        MarkRunType.TEST_MARK_REGRESSION_WEEKLY_SINGLE: 5 * 60,
        MarkRunType.TEST_MARK_ENABLING: 10 * 60,
        MarkRunType.TEST_MARK_STRESS_AND_LOAD: 40 * 60,
        MarkRunType.TEST_MARK_STRESS_AND_LOAD_SINGLE: 40 * 60,
        MarkRunType.TEST_MARK_LONG: 48 * 60 * 60,
        MarkRunType.TEST_MARK_SMOKE: 5 * 60,
        MarkRunType.TEST_MARK_MANUAL: 5 * 60,
        MarkRunType.TEST_MARK_PERFORMANCE: performance_test_timeout_minutes * 60,
    },
)

TIMEOUT_MULTIPLIER: dict = {
    TargetDevice.GPU: 1.5,
    TargetDevice.NPU: 1.5,
    "TRACE_TOOLS": 2,
    "AUTO_HETERO_MULTI": 3,
}

CURRENT_TARGET_DEVICE_DICT = {}

DEVICE_ID_TO_DETAILED_TARGET_DEVICE_NAME_MAP = defaultdict(lambda: ("", []), {})

SkippedItem = namedtuple("SkippedItem", "test_name reason")


def init_environment(_config):
    global CURRENT_TARGET_DEVICE_DICT
    # additional constant CURRENT_TARGET_DEVICE_DICT needs to be used due to being unable to read
    # current_target_device_dict from config when using xdist=0
    _config.current_target_device_dict = CURRENT_TARGET_DEVICE_DICT

    if not machine_is_reserved_for_test_session:
        return
    init_cleanup()


def init_cleanup():
    if cleanup_env_on_startup:
        if get_host_os() == OsType.Windows:
            cleanup_ovms_processes()
        else:
            cleanup_docker(cleanup_docker_containers)


def clean_container(container):
    try:
        container.stop(timeout=1)
        container.remove(force=True)
    except docker_errors.NotFound:
        logger.warning(f"Container: {container.name} already removed")
    except docker_errors.APIError:
        logger.warning(f"Removal of container: {container.name} already in progress")
    else:
        logger.warning(f"Killing running container: {container.name}")


def cleanup_docker(cleanup_docker_func):
    try:
        cleanup_docker_func()
    except docker_errors.APIError as error:
        logger.warning(f"Error occurred during docker cleanup: {error}")


def cleanup_docker_containers():
    dc = DockerContainer(None)
    for container in dc.list_containers():
        clean_container(container)
    logger.warning("Removing all stopped containers")
    prune_results = dc.prune()
    containers_deleted = prune_results.get("ContainersDeleted", [])
    for container in containers_deleted or []:
        logger.info(f"Removed container: {str(container)}")


def cleanup_docker_images():
    """Remove docker images build during test session"""
    if OsType.Windows in config.base_os:
        return
    docker_client = DockerClient()
    test_object_prefix = get_test_object_prefix()
    for image in docker_client.images.list():
        for image_tag in image.tags:
            if test_object_prefix in image_tag:
                docker_client.images.remove(image=image.id, force=True, noprune=False)
                logger.info(f"Removed docker image: {image.id}")


def cleanup_tmp_repos_dir(config):
    try:
        shutil.rmtree(config.tmp_repos_dir)
    except PermissionError as e:
        if get_host_os() == OsType.Windows and type(e) == PermissionError:
            change_dir_permissions(config.tmp_repos_dir)
            shutil.rmtree(config.tmp_repos_dir)


def teardown_environment():
    if get_host_os() == OsType.Windows:
        if config.teardown_ovms_processes:
            cleanup_ovms_processes()
    else:
        if config.teardown_docker_containers:
            cleanup_docker(cleanup_docker_containers)
        if config.teardown_docker_images:
            cleanup_docker(cleanup_docker_images)


def cleanup_ovms_processes():
    proc = Process()
    proc.disable_check_stderr()
    proc.run("taskkill /F /IM ovms.exe /T", print_stdout=False)


def clear_ovms_capi_artifacts():
    if not (cleanup_env_on_startup and machine_is_reserved_for_test_session):
        return
    proc = Process()
    proc.disable_check_stderr()
    if get_host_os() == OsType.Windows:
        if os.path.exists(c_api_wrapper_dir):
            proc.run_and_check(f"rmdir /S /Q {c_api_wrapper_dir}")
    else:
        proc.run_and_check("make clean", cwd=Paths.OVMS_TEST_CAPI_WRAPPER_DIR)
        proc.run_and_check(f"sudo rm -rf {c_api_wrapper_dir}")


def setup_artifacts_dir():
    if not config.artifacts_dir:
        return
    artifacts_dir_path = Path(config.artifacts_dir)
    if not artifacts_dir_path.exists():
        artifacts_dir_path.mkdir(parents=True)

    if config.clean_artifacts_dir:
        for file in artifacts_dir_path.glob("*"):
            logger.info(f"Deleting old artifacts: {file}")
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()


def setup_capi_wrapper(package_content):
    if OvmsType.CAPI not in config.ovms_types:
        return
    for _src, _dst in [
        (Paths.OVMS_TEST_CAPI_WRAPPER_PYX, Path(package_content, "include")),
        (Paths.OVMS_TEST_CAPI_WRAPPER_MAKEFILE, package_content),
        (Paths.OVMS_TEST_CAPI_WRAPPER_SETUP, package_content),
        (Paths.OVMS_TEST_CAPI_AUTOPXD_PY, Path(package_content, "include")),
    ]:
        shutil.copy(_src, _dst)

    proc = Process()
    proc.disable_check_stderr()

    # Example:
    # >>> sys.executable
    # '/usr/local/ovms-test/.venv/bin/python3'
    # >>> venv_path
    # '/usr/local/ovms-test/.venv'
    venv_path = str(Path(*Path(sys.executable).parts[:-2]))
    _stdout = proc.run_and_check(f"PYVENV={venv_path} make", cwd=package_content)


def download_binary_package(binary_package_src_file_path, binary_package_dst_file_path):
    wget_file(binary_package_src_file_path, binary_package_dst_file_path)


def get_binary_artifacts(
    binary_package_src_file_path, binary_package_dst_path, ovms_binary_name=OVMS_BINARY_PACKAGE_NAME
):
    proc = Process()
    proc.disable_check_stderr()
    print(f"Preparing OVMS package: {binary_package_src_file_path}")
    used_extensions = [extension for extension in OVMS_BINARY_PACKAGE_EXTENSIONS
                       if binary_package_src_file_path.endswith(extension)]
    if not used_extensions:
        raise NotImplementedError(
            f"OVMS binary supported only with .tar.gz or .zip formats. "
            f"Current package name: {binary_package_src_file_path}"
        )
    ovms_binary_full_name = f"{ovms_binary_name}{used_extensions[0]}"

    if binary_package_src_file_path.startswith("http"):
        binary_package_dst_file_path = os.path.join(binary_package_dst_path, ovms_binary_full_name)
        download_binary_package(binary_package_src_file_path, binary_package_dst_file_path)
    else:
        ovms_binary_src_path = os.path.realpath(os.path.expanduser(binary_package_src_file_path))
        if not os.path.exists(binary_package_dst_path):
            os.makedirs(binary_package_dst_path)
        shutil.copy(ovms_binary_src_path, os.path.join(binary_package_dst_path, ovms_binary_full_name))
    proc.run_and_check(f"tar -xf {ovms_binary_full_name}", cwd=binary_package_dst_path)
    setupvars_script_dst = os.path.join(binary_package_dst_path, "ovms", "setupvars.bat")
    if not os.path.exists(setupvars_script_dst):
        shutil.copy2(config.setupvars_script_path, setupvars_script_dst)


def run_docker_build_ovms_image(cmd, ovms_image_name, cwd, timeout=None):
    print(f"Building {ovms_image_name} image using cmd: {cmd}")
    proc = Process()
    proc.disable_check_stderr()
    code, stdout, stderr = proc.run_and_check_return_all(cmd, cwd=cwd, timeout=timeout)
    assert (f"naming to {ovms_image_name}" in stderr) or (
        f"Successfully tagged {ovms_image_name}" in stdout
    ), f"Image was not built successfully; stderr: {stderr}"
    print(f"Ovms-test image {ovms_image_name} successfully created")


def get_ovms_capi_docker_build_cmd(ovms_image, base_os, dockerfile, ovms_binary_image_name):
    base_image = config.base_image if config.base_image else os_type_to_base_image_binary_docker[base_os]
    ovms_test_image = calculate_ovms_test_image_name(ovms_image) if build_test_image else base_image
    target_device = TargetDevice.GPU if TargetDevice.GPU.lower() in ovms_image else TargetDevice.CPU
    cmd = (
        f"docker build -f {dockerfile} -t {ovms_binary_image_name} . "
        f"--build-arg BASE_IMAGE={base_image} "
        f"--build-arg OVMS_IMAGE={ovms_image} "
        f"--build-arg OVMS_TEST_IMAGE={ovms_test_image} "
        f"--build-arg OVMS_DEPENDENCIES='{OVMS_CAPI_DEPENDENCIES[base_os]}' "
        f"--build-arg TOOLS_DEPENDENCIES='{OVMS_CAPI_TOOLS_DEPENDENCIES[target_device][base_os]}' "
        f"--build-arg INSTALL_DRIVER_VERSION='{GPU_INSTALL_DRIVER_VERSION[base_os]}' "
        f"--build-arg http_proxy={http_proxy} "
        f"--build-arg https_proxy={https_proxy} "
        f"--build-arg no_proxy={no_proxy} "
    )
    return cmd


def get_ovms_binary_docker_build_cmd(ovms_image, base_os, dockerfile, ovms_binary_image_name):
    base_image = config.base_image if config.base_image else os_type_to_base_image_binary_docker[base_os]
    ovms_test_image = calculate_ovms_test_image_name(ovms_image) if build_test_image else base_image
    cpu_extensions_path = Paths.ROOT_PATH_CPU_EXTENSIONS if build_test_image else DOCKER_CONTAINER_TMP_PATH
    custom_loader_path = Paths.CUSTOM_LOADER_LIBRARIES_PATH_INTERNAL if build_test_image else DOCKER_CONTAINER_TMP_PATH
    custom_nodes_path = Paths.CUSTOM_NODE_LIBRARIES_PATH_INTERNAL if build_test_image else DOCKER_CONTAINER_TMP_PATH
    cmd = (
        f"docker build -f {dockerfile} -t {ovms_binary_image_name} . "
        f"--build-arg BASE_IMAGE={base_image} "
        f"--build-arg OVMS_IMAGE={ovms_image} "
        f"--build-arg OVMS_TEST_IMAGE={ovms_test_image} "
        f"--build-arg OVMS_DEPENDENCIES='{OVMS_BINARY_DEPENDENCIES[base_os]}' "
        f"--build-arg CPU_EXTENSIONS_PATH={cpu_extensions_path} "
        f"--build-arg CUSTOM_LOADER_PATH={custom_loader_path} "
        f"--build-arg CUSTOM_NODES_PATH={custom_nodes_path} "
        f"--build-arg http_proxy={http_proxy} "
        f"--build-arg https_proxy={https_proxy} "
        f"--build-arg no_proxy={no_proxy} "
    )
    return cmd


def build_ovms_binary_image():
    ovms_c_artifacts = {_base_os: config.ovms_c_release_artifacts_path[index] for index, _base_os in enumerate(config.base_os)}

    for ovms_image, base_os in get_ovms_calculated_images():
        ovms_binary_dst_path = os.path.join(tmp_dir, "ovms_binary", base_os)
        get_binary_artifacts(ovms_c_artifacts[base_os], ovms_binary_dst_path)

        dockerfile = f"Dockerfile.{UBUNTU if UBUNTU in base_os else base_os}"
        shutil.copy(
            os.path.join(os.path.dirname(__file__), "ovms_binary_image", dockerfile),
            ovms_binary_dst_path,
        )

        ovms_binary_image_name = calculate_ovms_binary_image_name(ovms_image)
        cmd = get_ovms_binary_docker_build_cmd(ovms_image, base_os, dockerfile, ovms_binary_image_name)
        run_docker_build_ovms_image(cmd, ovms_binary_image_name, cwd=ovms_binary_dst_path, timeout=None)


def build_ovms_capi_image():
    ovms_c_artifacts = {_base_os: config.ovms_c_release_artifacts_path[index] for index, _base_os in enumerate(config.base_os)}

    for ovms_image, base_os in get_ovms_calculated_images():
        ovms_capi_dst_path = os.path.join(tmp_dir, "ovms_capi", base_os)
        get_binary_artifacts(ovms_c_artifacts[base_os], ovms_capi_dst_path)
        if TargetDevice.GPU.lower() in ovms_image:
            for gpu_install_script in GPU_INSTALL_SCRIPTS[base_os]:
                shutil.copy(os.path.join(ovms_c_repo_path, gpu_install_script), ovms_capi_dst_path)
        else:
            for gpu_install_script in GPU_INSTALL_SCRIPTS[base_os]:
                with open(os.path.join(ovms_capi_dst_path, gpu_install_script), "a"):
                    pass

        dockerfile = f"Dockerfile.{UBUNTU if UBUNTU in base_os else base_os}"
        shutil.copy(
            os.path.join(os.path.dirname(__file__), "ovms_capi_image", dockerfile),
            ovms_capi_dst_path,
        )

        ovms_capi_image_name = calculate_ovms_capi_image_name(ovms_image)
        cmd = get_ovms_capi_docker_build_cmd(ovms_image, base_os, dockerfile, ovms_capi_image_name)
        run_docker_build_ovms_image(cmd, ovms_capi_image_name, cwd=ovms_capi_dst_path, timeout=None)


def prepare_ovms_package():
    if all([
        all([OvmsType.CAPI not in ovms_type for ovms_type in config.ovms_types]),
        all([OvmsType.BINARY not in ovms_type for ovms_type in config.ovms_types]),
    ]):
        return

    if any([OvmsType.CAPI in config.ovms_types, OvmsType.BINARY in config.ovms_types]):
        assert (
            len(config.base_os) == 1 and get_host_os() == config.base_os[0]
        ), f"Mismatch between config base os: {config.base_os}; host os: {get_host_os()}"

    for base_os in config.base_os:
        ovms_binary_dst_path = os.path.join(c_api_wrapper_dir, base_os)
        get_binary_artifacts(config.ovms_c_release_artifacts_path[0], ovms_binary_dst_path)

        package_content = Path(Paths.CAPI_WRAPPER_PACKAGE_CONTENT_PATH(base_os))
        setup_capi_wrapper(package_content)


def get_models_to_download():
    models_to_download = []
    for various_models_name in [name for name, obj in vars(ModelsLibrary).items() if isinstance(obj, property)]:
        various_models_value = getattr(ModelsLib, various_models_name)
        if isinstance(various_models_value, dict):
            for target_device in target_devices:
                models_to_download.extend(various_models_value[target_device])
        else:
            models_to_download.extend(various_models_value)
    return list(set(models_to_download))


def download_models():
    models_to_download = get_models_to_download()
    for model_type in models_to_download:
        if model_type.is_local:
            ov_hf_downloader = OVHfDownloader(model_type)
            ov_hf_downloader.check_and_update_hf_model()


def get_docker_images(images_to_download):
    images_to_download.add(config.minio_image)
    for target_device, base_os in itertools.product(config.target_devices, config.base_os):
        ovms_image = calculate_ovms_image_name(target_device, base_os)
        if config.ovms_image_local:
            OvmsInfo.get_local_image(ovms_image)
        else:
            images_to_download.add(ovms_image)
    return images_to_download


def download_docker_images():
    docker_ovms_types = [
        OvmsType.DOCKER, OvmsType.DOCKER_CMD_LINE, OvmsType.BINARY_DOCKER, OvmsType.CAPI_DOCKER
    ]
    if not any(_ovms_type in docker_ovms_types for _ovms_type in config.ovms_types):
        return

    images_to_download = set()
    images_to_download = get_docker_images(images_to_download)

    for image in images_to_download:
        if image:
            OvmsInfo.pull_latest_image(image)


def download_resources_master():
    print("Download required resources")
    download_models()
    download_docker_images()


def init_ovms_config_retrieved_from_master(pytest_config):
    config.tmp_repos_dir = pytest_config.workerinput[TMP_REPOS_DIR_ARGUMENT]
    global CURRENT_TARGET_DEVICE_DICT
    CURRENT_TARGET_DEVICE_DICT = pytest_config.workerinput[CURRENT_TARGET_DEVICE_DICT_ARGUMENT]


def build_local_resources():
    if OvmsType.BINARY_DOCKER in config.ovms_types:
        build_ovms_binary_image()
    if OvmsType.CAPI_DOCKER in config.ovms_types:
        build_ovms_capi_image()


def setup_tmp_repos_dir(config):
    config.tmp_repos_dir = TmpDir()


def get_marker_args(metafunc, marker_name):
    _marker = [marker for marker in metafunc.definition.own_markers if marker.name == marker_name]
    if _marker:
        return _marker[0].args
    return


def get_ids_with_target_device(parameter, func):
    # return id for target_device
    if parameter in vars(TargetDevice).values() or (isinstance(parameter, str) and get_base_device(parameter)
                                                    in vars(TargetDevice).values()):
        return CURRENT_TARGET_DEVICE_DICT.get(parameter, parameter)
    # return custom id
    return func(parameter)


def parametrize_model_type(metafunc):
    args = get_marker_args(metafunc, MarkTestParameters.MODEL_TYPE)
    if args is None:
        parametrize_target_device(metafunc)
        return
    if isinstance(args[0], dict):
        params_list = [
            (device_type, result) for device_type in config.target_devices
            for result in args[0][get_base_device(device_type)]
        ]
    else:
        params_list = [(device_type, result) for device_type in config.target_devices for result in args[0]]
    ids_list = [
        f"{CURRENT_TARGET_DEVICE_DICT.get(device_type, device_type)}-{model_type.__name__}"
        for device_type, model_type in params_list
    ]
    metafunc.parametrize(f"{TARGET_DEVICE_PARAM_NAME}, {MarkTestParameters.MODEL_TYPE}", params_list, ids=ids_list)


def parametrize_model_aux_type(metafunc):
    """Parametrize auxiliary (second) model for tests that need two models simultaneously.

    Requires model_type to also be present in the test — model_type handles target_device parametrization,
    so model_aux_type only parametrizes the model class itself (no device cross-product).
    """
    assert MarkTestParameters.MODEL_TYPE in metafunc.fixturenames, (
        f"model_aux_type requires model_type to also be a fixture in test {metafunc.function.__name__}"
    )
    args = get_marker_args(metafunc, MarkTestParameters.MODEL_AUX_TYPE)
    if args is None:
        return
    if isinstance(args[0], dict):
        params_list = [model for device_type in config.target_devices for model in args[0][get_base_device(device_type)]]
    else:
        params_list = list(args[0])
    ids_list = [model.__name__ for model in params_list]
    metafunc.parametrize(MarkTestParameters.MODEL_AUX_TYPE, params_list, ids=ids_list)


def parametrize_all_models(metafunc):
    args = get_marker_args(metafunc, MarkTestParameters.ALL_MODELS)
    if args is None:
        parametrize_target_device(metafunc, config.target_devices)
        return
    params_list = []
    for device_type in config.target_devices:
        for _models in args[0]:
            if isinstance(_models, dict):
                params_list.append((device_type, _models[get_base_device(device_type)]))
            else:
                params_list.append((device_type, _models))
    ids_list = lambda i: get_ids_with_target_device(
        i, lambda x: x[-1].name if len(x) > 0 and hasattr(x[-1], "name") else "empty"
    )
    metafunc.parametrize(f"{TARGET_DEVICE_PARAM_NAME}, {MarkTestParameters.ALL_MODELS}", params_list, ids=ids_list)


def parametrize_many_models(metafunc):
    args = get_marker_args(metafunc, MarkTestParameters.MANY_MODELS)
    if args is None:
        parametrize_target_device(metafunc)
        return
    # (ModelsLib.get_many_models, 8, 100) -> ("CPU", (ModelsLib.get_many_models(8), 100))
    params_list = [
        (device_type, (arg[0](device_type, arg[1]), arg[2])) for device_type in config.target_devices for arg in args
    ]
    ids_list = lambda i: get_ids_with_target_device(i, lambda x: f"count={len(x[0])}-iters={x[1]}")
    metafunc.parametrize(f"{TARGET_DEVICE_PARAM_NAME}, {MarkTestParameters.MANY_MODELS}", params_list, ids=ids_list)


def parametrize_iteration_info(metafunc):
    args = get_marker_args(metafunc, MarkTestParameters.ITERATION_INFO)
    if args is None:
        parametrize_target_device(metafunc)
        return
    # [(ModelsLib.various_models, 0, False, True)] -> ("CPU", ([(ModelsLib.various_models["CPU"][0], False, True)])
    params_list = []
    for device_type in config.target_devices:
        for _iteration_info_item in args:
            iteration_info = []
            for _iteration_info in _iteration_info_item:
                iteration_info.append(
                    (_iteration_info[0][device_type][_iteration_info[1]], _iteration_info[2], _iteration_info[3])
                )
            params_list.append((device_type, iteration_info))
    ids_list = lambda i: get_ids_with_target_device(i, ModelsLib.generate_ids_for_iteration_info)
    metafunc.parametrize(f"{TARGET_DEVICE_PARAM_NAME}, {MarkTestParameters.ITERATION_INFO}", params_list, ids=ids_list)


def parametrize_input_shape(metafunc):
    params_list = []
    args = get_marker_args(metafunc, MarkTestParameters.INPUT_SHAPE)
    if args is not None:
        # (ModelsLib.create_input_shapes_for_auto_reshape_tests, ModelsLib.reshapeable_model, ModelType.ONNX) ->
        # ("CPU", ModelsLib.create_input_shapes_for_auto_reshape_tests(ModelsLib.reshapeable_model["CPU"][ModelType.ONNX]))
        params_list = [
            (device_type, shape)
            for device_type in config.target_devices
            for shape in args[0](args[1][device_type][args[2]])
        ]
    args = get_marker_args(metafunc, MarkTestParameters.INPUT_SHAPE_NO_AUTO)
    if args is not None:
        params_list = [(device_type, shape) for device_type in config.target_devices for shape in args[0][device_type]]
    if params_list:
        ids_list = lambda i: get_ids_with_target_device(i, ModelsLib.generate_model_shape_ids)
        metafunc.parametrize(f"{TARGET_DEVICE_PARAM_NAME}, {MarkTestParameters.INPUT_SHAPE}", params_list, ids=ids_list)
    else:
        parametrize_target_device(metafunc)


def parametrize_plugin_config(metafunc):
    args = get_marker_args(metafunc, MarkTestParameters.PLUGIN_CONFIG)
    if args is None:
        parametrize_target_device(metafunc)
        return
    params_list = [
        (device_type, plugin_config) for device_type in config.target_devices
        for plugin_config in args[0][get_base_device(device_type)]
    ]
    ids_list = lambda i: get_ids_with_target_device(i, lambda x: "-".join(map(lambda y: "%s=%s" % y, x.items())))
    metafunc.parametrize(f"{TARGET_DEVICE_PARAM_NAME}, {MarkTestParameters.PLUGIN_CONFIG}", params_list, ids=ids_list)


def parametrize_target_device(metafunc):
    ids = [CURRENT_TARGET_DEVICE_DICT.get(x, x) for x in config.target_devices]
    metafunc.parametrize(TARGET_DEVICE_PARAM_NAME, config.target_devices, ids=ids)


def validate_lock_files():
    """Ensure that target_device locks files exists"""
    if not machine_is_reserved_for_test_session:
        return  # Cannot validate locks validity since other testing session could acquire device lock

    locks = [value for key, value in vars(Paths).items() if "LOCK_FILE" in key]
    for target_device in config.target_devices:
        n = MAX_WORKERS_PER_TARGET_DEVICE[get_base_device(target_device)]
        locks += [Paths.get_target_device_lock_file(target_device, i) for i in range(n)]
    for lock_path in [Path(x) for x in locks]:
        if lock_path.exists():
            logger.warning(f"Hanging lock file discovered:\n{lock_path.name}")
            logger.warning(f"Deleting lock file:\n{lock_path.name}")
            lock_path.unlink()


def list_host_zombie_processes():
    zombie_pids = []
    if OsType.Windows in config.base_os:
        return zombie_pids
    all_pids = [x.name for x in Path("/proc").iterdir() if str(x.name).isnumeric()]
    zombie_pids = [x for x in all_pids if get_pid_status(x) == PID_STATE_ZOMBIE]
    if len(zombie_pids) > 0:
        logger.warning(f"Found {len(zombie_pids)} zombie processes.")
        for zombie in zombie_pids:
            logger.warning(f"Zombie:\t{get_pid_name(zombie)}")
    return zombie_pids


def parametrize_ovms_type(metafunc):
    metafunc.parametrize(OVMS_TYPE_PARAM_NAME, config.ovms_types)


def parametrize_uses_mapping(metafunc):
    value_to_id = {
        True: "use_mapping",
        False: "no_mapping",
        None: "default_model_mapping",
    }
    ids = [value_to_id[x] for x in config.uses_mapping]
    metafunc.parametrize(USES_MAPPING_PARAM_NAME, config.uses_mapping, ids=ids)


def validate_port_pool(_config):
    # This function should be called only in master xdist thread.
    if any([
        config.ports_pool_size is None,
        config.grpc_ovms_starting_port is None,
        config.rest_ovms_starting_port is None,
    ]):
        print("Creating pool configuration:")
        args = parse_args([
            "--reservation-file-json",
            os.path.join(tmp_dir, "reservation.json"),
            "--reservation-file-env",
            os.path.join(tmp_dir, "reservation.env"),
            "-c",
            os.path.join(ovms_c_repo_path, "tests", "reservation_manager.yml"),
            "--locks-dir",
            global_tmp_dir_default,
        ])

        _config.reservation_manager = ReservationManager.manager_from_args(args=args)
        _config.reservation_manager.independent.create()
        os.environ.update(_config.reservation_manager.env_mgr.environment)
        config.ports_pool_size = config.get_int("TT_PORTS_POOL_SIZE")
        config.grpc_ovms_starting_port = config.get_int("TT_GRPC_OVMS_STARTING_PORT")
        config.rest_ovms_starting_port = config.get_int("TT_REST_OVMS_STARTING_PORT")
    else:
        print("Successfully read reservation manager configuration.")
    print("Port pool configuration:")
    print(f"ports_pool_size={config.ports_pool_size}")
    print(f"grpc_ovms_starting_port={config.grpc_ovms_starting_port}")
    print(f"rest_ovms_starting_port={config.rest_ovms_starting_port}")


def parametrize_base_os(metafunc):
    ids = []
    params = config.base_os
    for _os in config.base_os:
        if OsType.Windows in config.base_os:
            os_name = OsType.Windows
            if len(config.base_os) > 1:
                raise NotImplementedError("Iterating with Windows OS is not supported")
        elif any(_os.lower() == value for key, value in vars(OsType).items() if not key.startswith("__")):
            image = calculate_ovms_image_name(config.target_devices[0], _os)
            env_info = EnvironmentInfo.get_instance(class_info=OvmsInfo, image=image)
            dist_name = env_info.get_os_distname()
            if dist_name.startswith("Ubuntu 22.04") or dist_name.startswith(OsType.Ubuntu22):
                os_name = OsType.Ubuntu22
            elif dist_name.startswith("Ubuntu 24.04") or dist_name.startswith(OsType.Ubuntu24):
                os_name = OsType.Ubuntu24
            elif dist_name.startswith("Red Hat") or dist_name.startswith(OsType.Redhat):
                os_name = OsType.Redhat
        else:
            raise Exception("Unexpected OS")
        if config.is_nginx_mtls:
            os_name = f"{os_name}_NGINX"
        ids.append(os_name.upper())
    metafunc.parametrize(BASE_OS_PARAM_NAME, params, ids=ids)


def log_configuration_variables():
    logger.info("============== configuration variables ==============")
    pt_env_vars = list(filter(lambda x: x[0].startswith("TT_"), os.environ.items()))
    pt_env_vars.sort()
    for env_var in pt_env_vars:
        logger.info("{}={}".format(*env_var))


def mute_warnings():
    # Mute warning:
    # ResourceWarning: unclosed <socket.socket fd=17, family=AddressFamily.AF_UNIX ...
    warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    #  .venv/lib/python3.10/site-packages/flatbuffers/compat.py:19:
    #  DeprecationWarning: the imp module is deprecated in favour of importlib;
    #  see the module's documentation for alternative uses import imp
    warnings.filterwarnings(
        action="ignore", message="the imp module is deprecated in favour of importlib", category=DeprecationWarning
    )

    # .venv/lib/python3.10/site-packages/tensorflow/python/framework/dtypes.py:205
    #   python3.10/site-packages/tensorflow/python/framework/dtypes.py:205:
    #   DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)
    #     np.bool8: (False, True),
    warnings.filterwarnings(
        action="ignore", message="`np.bool8` is a deprecated alias for `np.bool_`", category=DeprecationWarning
    )


def setup_nginx():
    if not is_nginx_mtls:
        return
    print("Setup nginx certificates")
    if is_xdist_master():
        OvsaCerts.generate_ovsa_certs(skip_if_valid=not force_generate_new_ssl_certs)
    OvsaCerts.init_ovsa_certs()


def remove_ports_reservation(_config):
    reservation_manager = getattr(_config, "reservation_manager", None)
    if reservation_manager is None:
        return
    logger.info("Removing self reserved ports")
    reservation_manager.independent.remove()

    if machine_is_reserved_for_test_session:
        # If machine is reserved, no other test session should active
        # So clean dangling reservations (if any occurs during previous fatal errors).
        # If machine never be reserved exclusively (ie. builder0x)
        # you can clean dangling reservation manually by deleting files:
        # `/tmp/reservation_manager-*-*-independent`
        # (after ensuring no test session is active)
        reservation_manager.independent.cleanup()


def clear_lockfiles():
    if not Path(ovms_file_locks_dir).exists():
        return
    for file in Path(ovms_file_locks_dir).iterdir():
        print(f"Delete hanging lock: {str(file)}")  # logger could be unavailable by now
        file.unlink()


def deselect_items(items, config, deselected):
    config.hook.pytest_deselected(items=deselected)
    for item in deselected:
        test_name = item.parent.nodeid
        # nodeid comes in a way:
        # 1) test.py::TestClass::()
        # 2) test.py::
        if test_name[-2:] == "()":
            test_name = test_name[:-2]
        elif "::" not in test_name:
            test_name += "::"

        test_name += item.name
        logger.debug("Deselecting test: " + test_name)
        items.remove(item)


def set_divide_target_device_per_worker(items):
    # Assign xdist_group per target_device so --dist loadgroup routes
    # all tests for a given device to the same worker.
    # Must be done before yield so xdist sees the markers during scheduling.
    if config.divide_target_device_per_worker:
        num_devices = len(config.target_devices)
        if num_devices and config.xdist_workers > 0 and config.xdist_workers % num_devices != 0:
            raise ValueError(
                f"xdist_workers ({config.xdist_workers}) must be a multiple of "
                f"the number of target devices ({num_devices}): {config.target_devices}"
            )
        for item in items:
            if hasattr(item, "callspec") and TARGET_DEVICE_PARAM_NAME in item.callspec.params:
                td = item.callspec.params[TARGET_DEVICE_PARAM_NAME]
                item.add_marker(pytest.mark.xdist_group(name=f"device_{td}"))
                logger.debug(f"Assigned {item.nodeid} to xdist_group device_{td}")


def preprocess_collected_items(items):
    deselected = []
    all_components = {}
    all_requirements = {}
    try:
        required_marker_ids, excluded_marker_ids = get_marker_ids_for_test_run()
        for item in items:
            set_item_image_parameter(item)
            preprocess_collected_item(
                item,
                deselected,
                all_components,
                all_requirements,
                required_marker_ids,
                excluded_marker_ids,
            )

    except RuntimeError as e:
        error_msg = str(e)
        logger.exception(error_msg)
        sys.exit(error_msg)

    return deselected


def get_marker_ids_for_test_run():
    # requirements
    if req_ids and exclude_req_ids:
        raise RuntimeError("Can't both include and exclude requirements!")
    # components
    if components_ids and exclude_components_ids:
        raise RuntimeError("Can't both include and exclude components!")

    required_marker_ids = generate_marker_ids(req_ids, components_ids, tests_priority_list)
    excluded_marker_ids = generate_marker_ids(exclude_req_ids, exclude_components_ids)
    return required_marker_ids, excluded_marker_ids


def generate_marker_ids(*args):
    ids_lists = [ids_list for ids_list in args if ids_list]
    marker_ids = []
    if len(ids_lists) > 1:
        marker_ids = list(itertools.product(*ids_lists))
    elif len(ids_lists) == 1:
        marker_ids = [(id_value,) for id_value in ids_lists[0]]
    return marker_ids


def preprocess_collected_item(
        item, deselected, all_components, all_requirements, required_marker_ids, excluded_marker_ids
):
    apply_conditional_run_type_marks(item)
    test_type = MarkRunType.get_test_type_mark(item)
    set_timeout_per_test_type(item, test_type)
    update_parent_markers(
        item, (
            MarkGeneral.COMPONENTS.mark,
            MarkGeneral.REQIDS.mark,
            MarkPriority.HIGH.mark,
            MarkPriority.MEDIUM.mark,
            MarkPriority.LOW.mark,
        )
    )
    if deselect(item, test_type, required_marker_ids, excluded_marker_ids):
        deselected.append(item)
    else:
        update_markers(item, test_type, all_components, MarkGeneral.COMPONENTS.mark)
        update_markers(item, test_type, all_requirements, MarkGeneral.REQIDS.mark)
    return deselected


def set_item_image_parameter(item):
    if getattr(item, "callspec", None):
        # Store calculated image for later use.
        ovms_type = item.callspec.params.get(OVMS_TYPE_PARAM_NAME, OvmsType.DOCKER)
        base_os = item.callspec.params.get(BASE_OS_PARAM_NAME, OsType.Ubuntu22)
        if ovms_type == OvmsType.BINARY or ovms_type == OvmsType.CAPI:
            item._image = calculate_ovms_binary_name(base_os=base_os)
        else:
            target_device = item.callspec.params.get(TARGET_DEVICE_PARAM_NAME, TargetDevice.CPU)
            item._image = calculate_ovms_image_name(target_device=target_device, base_os=base_os)


def apply_conditional_run_type_marks(item):
    """Resolve conditional_run_type and conditional_run_type_by_model meta-markers.

    conditional_run_type: assigns single_mark when device+OS match, default_mark otherwise.
    conditional_run_type_by_model: assigns mark based on model_type membership in model collections.
    """
    params = getattr(getattr(item, 'callspec', None), 'params', {})

    for marker in item.iter_markers(MarkConditionalRunType.CONDITIONAL_RUN_TYPE):
        single_mark = marker.kwargs["single_mark"]
        default_mark = marker.kwargs["default_mark"]
        single_if_device = marker.kwargs.get("single_if_device")
        single_if_os = marker.kwargs.get("single_if_os")

        device = params.get(TARGET_DEVICE_PARAM_NAME, "")
        base_os = str(params.get(BASE_OS_PARAM_NAME, "")).lower()

        is_single = True
        if single_if_device and device not in single_if_device:
            is_single = False
        if single_if_os and base_os not in single_if_os:
            is_single = False

        mark_name = single_mark if is_single else default_mark
        item.add_marker(getattr(pytest.mark, mark_name))
        return  # only first conditional_run_type marker is applied

    for marker in item.iter_markers(MarkConditionalRunType.CONDITIONAL_RUN_TYPE_BY_MODEL):
        model_type = params.get(MarkTestParameters.MODEL_TYPE)
        if model_type is None:
            continue
        device = params.get(TARGET_DEVICE_PARAM_NAME, "")
        for mark_name, model_collection in marker.kwargs.get("model_mark_map", {}).items():
            device_models = set(model_collection.get(device, []))
            if model_type in device_models:
                item.add_marker(getattr(pytest.mark, mark_name))
                return
        default_mark = marker.kwargs.get("default_mark")
        if default_mark:
            item.add_marker(getattr(pytest.mark, default_mark))
        return


def set_timeout_per_test_type(item, test_type):
    if item.get_closest_marker("timeout") is None:
        value = timeout_dict[test_type]
        if any([test_type == MarkRunType.TEST_MARK_REGRESSION,
                test_type == MarkRunType.TEST_MARK_ON_COMMIT,
            ]):
            if any(["AUTO" in item.name, "HETERO" in item.name, "MULTI" in item.name]):
                value *= TIMEOUT_MULTIPLIER["AUTO_HETERO_MULTI"]
            elif TargetDevice.GPU in item.name:
                value *= TIMEOUT_MULTIPLIER[TargetDevice.GPU]
            elif TargetDevice.NPU in item.name:
                value *= TIMEOUT_MULTIPLIER[TargetDevice.NPU]
        if run_ovms_with_valgrind or run_ovms_with_opencl_trace:
            value *= TIMEOUT_MULTIPLIER["TRACE_TOOLS"]
        item.add_marker(pytest.mark.timeout(value))


def update_parent_markers(item, marker_types):
    for marker_type in marker_types:
        components = item.get_closest_marker(marker_type)
        if components is not None:
            current_components = next(
                (component for component in item.own_markers if component.name == marker_type),
                None,
            )
            if current_components is None:
                item.own_markers.append(components)


def deselect(item, test_type, required_marker_ids, excluded_marker_ids):
    # Validate different scenarios where test should be deselected from execution during `collect` stage.
    if isinstance(item, Function):
        if test_type is None:
            raise RuntimeError("Test do not have test_type: " + item.name)

        if required_marker_ids:
            for required_marker_id_list in required_marker_ids:
                if _is_test_marker_id_is_matched_with_id(item, required_marker_id_list):
                    # make sure that item is not deselected by other marker
                    return deselect_by_excluded_marker_ids(item, excluded_marker_ids)
            return True
        elif excluded_marker_ids:
            return deselect_by_excluded_marker_ids(item, excluded_marker_ids)

    return False


def deselect_by_excluded_marker_ids(item, excluded_marker_ids):
    for excluded_marker_ids_list in excluded_marker_ids:
        if _is_test_marker_id_is_matched_with_id(item, excluded_marker_ids_list):
            return True
    return False


def _is_test_marker_id_is_matched_with_id(test, ids_to_check: list):
    markers_to_check = []

    for marker in test.own_markers:
        if any([
            marker.name is MarkGeneral.REQIDS.value,
            marker.name is MarkGeneral.COMPONENTS.value,
            marker.name is MarkPriority.HIGH.mark,
            marker.name is MarkPriority.MEDIUM.mark,
            marker.name is MarkPriority.LOW.mark,
        ]):
            if marker.args:
                for marker_arg in marker.args:
                    if isinstance(marker_arg, dict):
                        for param in marker_arg:
                            if param is None:
                                markers_to_check.append(str(marker_arg.values))
                            elif param in test.name:
                                markers_to_check.append(str(marker_arg.values()))
                    elif isinstance(marker_arg, str):
                        markers_to_check.append(marker_arg)
                    else:
                        raise RuntimeError(
                            f"Test {test.name} do not have mark in correct form. Form: {type(marker_arg)}"
                        )
            else:
                markers_to_check.append(marker.name)

    check_list = []
    for id_to_check in ids_to_check:
        check_list.append(any(id_to_check.lower() in marker_to_check.lower() for marker_to_check in markers_to_check))

    return all(check_list)


def update_markers(item, test_type, markers, marker_type):
    marker = item.get_closest_marker(marker_type)
    if marker is not None:
        if test_type not in markers:
            markers[test_type] = set()
        markers[test_type].update(set(marker.args))


def get_skipped_items(items):
    skipped_items = [item for item in items if item.keywords.get("skip") is not None]
    items = []
    for item in skipped_items:
        skip_info = item.keywords.get("skip")
        if isinstance(skip_info, (Mark, MarkDecorator)):
            if "reason" in skip_info.kwargs:
                reason = skip_info.kwargs["reason"]
            elif skip_info.args:
                reason = skip_info.args[0]
            else:
                reason = ""
            items.append(SkippedItem(item.nodeid, reason))
    return items


def calc_statistics(items):
    skipped_items = get_skipped_items(items)
    issue_numbers = []
    other_tests = []
    for item in skipped_items:
        match = re.search(r"DPNG-\d+", item.reason)
        if match:
            issue_numbers.append(match.group(0))
        else:
            issue_numbers.append("others")
            other_tests.append(item)
    return Counter(issue_numbers), other_tests


def log_labeled_stats(issues):
    msg = ["Skipped tests statistic:"]
    issues_sorted_by_quantity = sorted(issues.items(), key=lambda i: i[1], reverse=True)
    for issue, quantity in issues_sorted_by_quantity:
        msg.append("{:>11}: {:>6}".format(issue, quantity))
    logger.info("\n".join(msg))


def log_others(other_items):
    msg = ["Skipped tests not labeled with issue:"]
    items_grouped_by_reason = groupby(other_items, key=lambda i: i.reason)
    for reason, items in list(items_grouped_by_reason):
        msg.append("{}:".format(reason))
        msg.extend("|---{}".format(item.test_name) for item in list(items))
    logger.info("\n".join(msg))


def log_skip_statistic(items):
    issue_stats, other_tests = calc_statistics(items)
    log_labeled_stats(issue_stats)
    log_others(other_tests)


def get_session_start_info(session):
    logger.info(f"Starting test session in the following folder: {session.startdir}")
    log_configuration_variables()
    session.start_time = time.time()


def parametrize_tests(metafunc):
    if OVMS_TYPE_PARAM_NAME in metafunc.fixturenames:
        parametrize_ovms_type(metafunc)

    if USES_MAPPING_PARAM_NAME in metafunc.fixturenames:
        parametrize_uses_mapping(metafunc)

    if BASE_OS_PARAM_NAME in metafunc.fixturenames:
        parametrize_base_os(metafunc)

    if MarkTestParameters.MODEL_TYPE in metafunc.fixturenames:
        parametrize_model_type(metafunc)
    elif MarkTestParameters.ALL_MODELS in metafunc.fixturenames:
        parametrize_all_models(metafunc)
    elif MarkTestParameters.MANY_MODELS in metafunc.fixturenames:
        parametrize_many_models(metafunc)
    elif MarkTestParameters.ITERATION_INFO in metafunc.fixturenames:
        parametrize_iteration_info(metafunc)
    elif MarkTestParameters.INPUT_SHAPE in metafunc.fixturenames:
        parametrize_input_shape(metafunc)
    elif MarkTestParameters.PLUGIN_CONFIG in metafunc.fixturenames:
        parametrize_plugin_config(metafunc)
    elif TARGET_DEVICE_PARAM_NAME in metafunc.fixturenames:
        parametrize_target_device(metafunc)

    if MarkTestParameters.MODEL_AUX_TYPE in metafunc.fixturenames:
        parametrize_model_aux_type(metafunc)
