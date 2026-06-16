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
import shutil
import sys
import warnings

from collections import defaultdict
from docker import errors as docker_errors
from pathlib import Path
from tests.functional import config
from ovms.constants.models_library import ModelsLib
from tests.functional.utils.download import wget_file
from tests.functional.utils.reservation_manager.args import parse_args
from tests.functional.utils.reservation_manager.manager import Manager as ReservationManager
from tests.functional.config import (
    build_test_image,
    c_api_wrapper_dir,
    cleanup_env_on_startup,
    global_tmp_dir_default,
    http_proxy,
    https_proxy,
    no_proxy,
    ovms_c_repo_path,
    machine_is_reserved_for_test_session,
    tmp_dir,
)
from tests.functional.constants.os_type import get_host_os, OsType, UBUNTU
from tests.functional.constants.os_version import os_type_to_base_image_binary_docker
from tests.functional.constants.ovms import (
    BASE_OS_PARAM_NAME,
    OVMS_TYPE_PARAM_NAME,
    TARGET_DEVICE_PARAM_NAME,
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
from tests.functional.object_model.ovms_info import OvmsInfo
from tests.functional.utils.core import TmpDir
from tests.functional.utils.docker import DockerClient, DockerContainer, DOCKER_CONTAINER_TMP_PATH
from tests.functional.utils.environment_info import EnvironmentInfo
from tests.functional.utils.logger import get_logger
from tests.functional.utils.marks import MarkTestParameters
from tests.functional.utils.process import PID_STATE_ZOMBIE, Process, get_pid_name, get_pid_status
from tests.functional.utils.test_framework import get_test_object_prefix

logger = get_logger(__name__)


CURRENT_TARGET_DEVICE_DICT = {}

DEVICE_ID_TO_DETAILED_TARGET_DEVICE_NAME_MAP = defaultdict(lambda: ("", []), {})


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
        proc.run_and_check(f"rm -rf {c_api_wrapper_dir}")


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


def download_resources_master():
    print("Download required resources")


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
    if parameter in vars(TargetDevice).values():
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
            (device_type, result) for device_type in config.target_devices for result in args[0][device_type]
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
        params_list = [model for device_type in config.target_devices for model in args[0][device_type]]
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
                params_list.append((device_type, _models[device_type]))
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
        (device_type, plugin_config) for device_type in config.target_devices for plugin_config in args[0][device_type]
    ]
    ids_list = lambda i: get_ids_with_target_device(i, lambda x: "-".join(map(lambda y: "%s=%s" % y, x.items())))
    metafunc.parametrize(f"{TARGET_DEVICE_PARAM_NAME}, {MarkTestParameters.PLUGIN_CONFIG}", params_list, ids=ids_list)


def parametrize_target_device(metafunc):
    ids = [CURRENT_TARGET_DEVICE_DICT.get(x, x) for x in config.target_devices]
    metafunc.parametrize(TARGET_DEVICE_PARAM_NAME, config.target_devices, ids=ids)


def validate_lock_files():
    """Ensure that target_device locks files exists"""
    if not config.machine_is_reserved_for_test_session:
        return  # Cannot validate locks validity since other testing session could acquire device lock

    locks = [value for key, value in vars(Paths).items() if "LOCK_FILE" in key]
    for target_device in config.target_devices:
        n = MAX_WORKERS_PER_TARGET_DEVICE[target_device]
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
