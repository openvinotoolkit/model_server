#
# Copyright (c) 2018-2026 Intel Corporation
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

import logging
import os
import sys
import time
from logging import FileHandler

import pytest
from _pytest._code import ExceptionInfo, filter_traceback  # noqa
from _pytest.outcomes import OutcomeException

from tests.functional.config import test_dir, test_dir_cleanup, artifacts_dir, enable_pytest_plugins
from tests.functional.constants.components import OvmsComponents
from tests.functional.constants.os_type import OsType
from tests.functional.constants.ovms import (
    BASE_OS_PARAM_NAME,
    OVMS_TYPE_PARAM_NAME,
    TARGET_DEVICE_PARAM_NAME,
    USES_MAPPING_PARAM_NAME,
)
from tests.functional.constants.ovms_binaries import calculate_ovms_binary_name
from tests.functional.constants.ovms_images import calculate_ovms_image_name
from tests.functional.constants.ovms_type import OvmsType
from tests.functional.constants.target_device import TargetDevice
from tests.functional.utils import hooks
from tests.functional.utils.hooks import (
    log_configuration_variables,
    parametrize_all_models,
    parametrize_base_os,
    parametrize_input_shape,
    parametrize_iteration_info,
    parametrize_many_models,
    parametrize_model_aux_type,
    parametrize_model_type,
    parametrize_ovms_type,
    parametrize_plugin_config,
    parametrize_target_device,
    parametrize_uses_mapping,
    validate_port_pool,
)
from tests.functional.utils.marks import MarksRegistry, MarkRunType, MarkTestParameters
from tests.functional.utils.test_framework import is_xdist_master

logger = logging.getLogger(__name__)


if enable_pytest_plugins:
    pytest_plugins = [
        "tests.functional.fixtures.ovms",
        "tests.functional.fixtures.server",
        "tests.functional.fixtures.api_type",
        "tests.functional.fixtures.params",
    ]


    def pytest_configure(config):
        """
        Allow plugins and conftest files to perform initial configuration.
        This hook is called for every plugin and initial conftest file after command line options have been parsed.
        After that, the hook is called for other conftest files as they are imported.

        NOTE:
            This hook is called multiple times:
            1) for master process prior spawning workers
                (PYTEST_XDIST_WORKER_COUNT and PYTEST_XDIST_WORKER env variable unset)
            2) for each spawned worker process

        LIMITATIONS:
            Internal pytest logging mechanisms are initialized in `pytest_sessionstart` hook.
            Please avoid usage of logger in all hooks used in this function.
            Please simple print(...) call for printing messages.
        """
        hooks.mute_warnings()
        MarksRegistry.register(config)

        if is_xdist_master():
            hooks.setup_tmp_repos_dir(config)
            validate_port_pool(config)
            # master thread pytest_configure call. No xdist worker process spawned yet.
            hooks.init_environment(config)
            hooks.clear_ovms_capi_artifacts()
            hooks.setup_artifacts_dir()
            hooks.prepare_ovms_package()
            hooks.download_resources_master()
            hooks.build_local_resources()
            hooks.validate_lock_files()
            hooks.list_host_zombie_processes()
        # else:  # Xdist worker thread
        #     hooks_utils.download_docker_images()
        #     hooks_utils.init_ovms_config_retrieved_from_master(config)

        # hooks_utils.setup_nginx()

        # Let know that pytest was successfully configured
        config.configured = True


    @pytest.hookimpl(hookwrapper=True)
    def pytest_collection_modifyitems(session, config, items):
        """
        Support for running tests with component tags.
        Report all test component markers to mongo_reporter.
        """
        logger.info("Preparing tests for test session in the following folder: {}".format(session.startdir))
        yield  # deselect items in default hook way via keyword ('-k')

        # if config.option.collectonly:
        #     hooks_utils.log_skip_statistic(items)

    #     deselected, all_components, all_requirements = preprocess_collected_items(items)

    #     if deselected:
    #         hooks_utils.deselect_items(items, config, deselected)

    #     random.Random(7).shuffle(items)


    def preprocess_collected_items(items):
        deselected = []
        all_components = {}
        all_requirements = {}
        # try:
    #         # required_marker_ids, excluded_marker_ids = get_marker_ids_for_test_run()
    #         for item in items:
    #             if getattr(item, "callspec", None):
    #                 # Store calculated image for later use.
    #                 ovms_type = item.callspec.params.get(OVMS_TYPE_PARAM_NAME, OvmsType.DOCKER)
    #                 base_os = item.callspec.params.get(BASE_OS_PARAM_NAME, OsType.Ubuntu22)
    #                 if ovms_type == OvmsType.BINARY or ovms_type == OvmsType.CAPI:
    #                     item._image = calculate_ovms_binary_name(base_os=base_os)
    #                 else:
    #                     target_device = item.callspec.params.get(
    #                         TARGET_DEVICE_PARAM_NAME,
    #                         TargetDevice.TARGET_DEVICE_CPU,
    #                     )
    #                     item._image = calculate_ovms_image_name(target_device=target_device, base_os=base_os)
    #             # add_dynamic_mark(item)
    #             test_type = MarkRunType.get_test_type_mark(item)
    #             # set_timeout_per_test_type(item, test_type)
    #             # update_parent_markers(item, MarkGeneral.COMPONENTS.mark)
    #             # update_parent_markers(item, MarkGeneral.REQIDS.mark)
    #             # update_parent_markers(item, MarkPriority.HIGH.mark)
    #             # update_parent_markers(item, MarkPriority.MEDIUM.mark)
    #             # update_parent_markers(item, MarkPriority.LOW.mark)
    #             # if deselect(item, test_type, required_marker_ids, excluded_marker_ids):
    #             #     deselected.append(item)
    #             #     continue
    #             # update_markers(item, test_type, all_components, MarkGeneral.COMPONENTS.mark)
    #             # update_markers(item, test_type, all_requirements, MarkGeneral.REQIDS.mark)
    #
    #     except RuntimeError as e:
    #         error_msg = str(e)
    #         logger.exception(error_msg)
    #         sys.exit(error_msg)
    #
    #     return deselected, all_components, all_requirements

    # def pytest_sessionfinish(session, exitstatus):
    #     current_test_run = hooks_utils.get_current_test_run()
    #     logger.info(
    #         "Finishing test session for test run type: {} in the following folder: {}".format(
    #             current_test_run, session.startdir
    #         )
    #     )
    #     test_status_report_header = f"{SEPARATOR} TEST TYPE STATUS REPORT - BEGIN {SEPARATOR}"
    #     logger.info(test_status_report_header)
    #
    #     data_to_save = hooks_utils.collect_test_status_data(exitstatus)
    #     hooks_utils.save_test_data(data_to_save)
    #
    #     test_status_report_footer = f"{SEPARATOR} TEST TYPE STATUS REPORT - END {SEPARATOR}"
    #     logger.info(test_status_report_footer)
    #     logger.info("Exit status is: {}".format(str(exitstatus)))
    #
    #     if current_test_run != "":
    #         hooks_utils.get_test_run_reporter(current_test_run).on_run_end()
    #     else:
    #         for reporter in test_run_reporters.values():  # Finish all reporters
    #             reporter.on_run_end()


    # def pytest_unconfigure(config):
    #     if getattr(config, "configured", None) is not True:
    #         # Check if pytest_configure() was done successfuly, if not: logger would be in invalid state so disable.
    #         for _logger in logger.manager.loggerDict.values():
    #             _logger.disabled = True
    #
    #     try:
    #         if is_xdist_master():
    #             if restler_generate_evidence:
    #                 restler_evidence_name_suffix = f"{release_product_version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    #                 shutil.make_archive(f"{restler_evidence_dir}_{restler_evidence_name_suffix}", "zip",
    #                                     restler_evidence_dir)
    #             hooks_utils.remove_ports_reservation(config)
    #             try:
    #                 shutil.rmtree(config.tmp_repos_dir)
    #             except PermissionError as e:
    #                 if all([
    #                     "C:\\" in config.tmp_repos_dir,
    #                     type(e) == PermissionError
    #                 ]):
    #                     # workaround for Windows: https://jira.devtools.intel.com/browse/CVS-161953
    #                     change_dir_permissions(config.tmp_repos_dir)
    #                     shutil.rmtree(config.tmp_repos_dir)
    #             hooks.teardown_environment()
    #             if machine_is_reserved_for_test_session:
    #                 hooks_utils.clear_lockfiles()
    #     except Exception as e:
    #         error_msg = str(e)
    #         print(error_msg)
    #         sys.exit(error_msg)

MarksRegistry.MARK_ENUMS.extend([OvmsComponents])


def pytest_sessionstart(session):
    logger.info("Starting test session in the following folder: {}".format(session.startdir))
    log_configuration_variables()
    session.start_time = time.time()


def pytest_generate_tests(metafunc):
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

    #pytest_runtest_protocol ? move

    #
    # def pytest_configure():
    #     # Perform initial configuration.
    #     init_logger()
    #
    #     init_conf_logger = logging.getLogger("init_conf")
    #
    #     container_names = get_containers_with_tests_suffix()
    #     if container_names:
    #         init_conf_logger.info("Possible conflicting container names: {} "
    #                               "for given tests_suffix: {}".format(container_names, get_tests_suffix()))
    #
    #     if artifacts_dir:
    #         os.makedirs(artifacts_dir, exist_ok=True)
    #
    #
    # def pytest_keyboard_interrupt(excinfo):
    #     clean_hanging_docker_resources()
    #     Server.stop_all_instances()
    #
    #
    # def pytest_unconfigure():
    #     # Perform cleanup.
    #     cleanup_logger = logging.getLogger("cleanup")
    #
    #     cleanup_logger.info("Cleaning hanging docker resources with suffix: {}".format(get_tests_suffix()))
    #     clean_hanging_docker_resources()
    #
    #     if test_dir_cleanup:
    #         cleanup_logger.info("Deleting test directory: {}".format(test_dir))
    #         delete_test_directory()
    #
    #     if len(Server.running_instances) > 0:
    #         logger.warning("Test got unstopped docker instances")
    #     Server.stop_all_instances()
    #
    #
    # @pytest.hookimpl(hookwrapper=True)
    # def pytest_collection_modifyitems(session, config, items):
    #     yield
    #     items = reorder_items_by_fixtures_used(session)
    #
    #
    # @pytest.hookimpl(tryfirst=True, hookwrapper=True)
    # def pytest_runtest_makereport(item, call):
    #     outcome = yield
    #     if call.when == "setup":
    #         report = outcome.get_result()
    #         report.test_metadata = {"start": call.start}
    #
    #
    # @pytest.hookimpl(hookwrapper=True)
    # def pytest_runtest_call():
    #     __tracebackhide__ = True
    #     try:
    #         outcome = yield
    #     finally:
    #         pass
    #     exception_catcher("call", outcome)
    #
    #
    # @pytest.hookimpl(hookwrapper=True)
    # def pytest_runtest_setup():
    #     __tracebackhide__ = True
    #     try:
    #         outcome = yield
    #     finally:
    #         pass
    #     exception_catcher("setup", outcome)
    #
    #
    # @pytest.hookimpl(hookwrapper=True)
    # def pytest_runtest_teardown():
    #     __tracebackhide__ = True
    #     try:
    #         outcome = yield
    #     finally:
    #         pass
    #     exception_catcher("teardown", outcome)
    #
    #
    # @pytest.hookimpl(hookwrapper=True)
    # def pytest_runtest_teardown(item):
    #     yield
    #     # Test finished: remove test item for all fixtures that was used
    #     for fixture in item._server_fixtures:
    #         if item in item.session._server_fixtures_to_tests[fixture]:
    #             item.session._server_fixtures_to_tests[fixture].remove(item)
    #         if len(item.session._server_fixtures_to_tests[fixture]) == 0:
    #             # No other tests will use this docker instance so we can close it.
    #             Server.stop_by_fixture_name(fixture)
    #
    #
    # def exception_catcher(when: str, outcome):
    #     if isinstance(outcome.excinfo, tuple):
    #         if len(outcome.excinfo) > 1 and isinstance(outcome.excinfo[1], OutcomeException):
    #             return
    #         exception_logger = logging.getLogger("exception_logger")
    #         exception_info = ExceptionInfo.from_exc_info(outcome.excinfo)
    #         exception_info.traceback = exception_info.traceback.filter(filter_traceback)
    #         exc_repr = exception_info.getrepr(style="short", chain=False)\
    #             if exception_info.traceback\
    #             else exception_info.exconly()
    #         exception_logger.error('Unhandled Exception during {}: \n{}'
    #                                .format(when.capitalize(), str(exc_repr)))
    #
    #
    # @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    # def pytest_runtest_logstart(nodeid, location):
    #     if artifacts_dir:
    #         test_name = get_path_friendly_test_name(location)
    #         log_path = os.path.join(artifacts_dir, f"{test_name}.log")
    #         _root_logger = logging.getLogger(None)
    #         _root_logger._test_log_handler = FileHandler(log_path)
    #         formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    #         _root_logger._test_log_handler.setFormatter(formatter)
    #         _root_logger.addHandler(_root_logger._test_log_handler)
    #     yield
    #
    #
    # @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    # def pytest_runtest_logfinish(nodeid, location):
    #     if artifacts_dir:
    #         _root_logger = logging.getLogger(None)
    #         _root_logger.removeHandler(_root_logger._test_log_handler)
    #     yield

