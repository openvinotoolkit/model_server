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

import random
import sys

from tests.functional.config import enable_pytest_plugins, pytest_keyword_filter, machine_is_reserved_for_test_session
from tests.functional.constants.ovms import (
    CURRENT_TARGET_DEVICE_DICT_ARGUMENT,
    TMP_REPOS_DIR_ARGUMENT,
)
from tests.functional.utils import hooks
from tests.functional.utils.logger import OvmsFileHandler, get_logger
from tests.functional.utils.marks import MarksRegistry
from tests.functional.utils.test_framework import is_xdist_master

logger = get_logger(__name__)


if enable_pytest_plugins:

    raise NotImplementedError("OVMS tests not enabled")

    pytest_plugins = [      # pylint: disable=unreachable
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
            hooks.validate_port_pool(config)
            # master thread pytest_configure call. No xdist worker process spawned yet.
            hooks.init_environment(config)
            hooks.clear_ovms_capi_artifacts()
            hooks.setup_artifacts_dir()
            hooks.prepare_ovms_package()
            hooks.download_resources_master()
            hooks.build_local_resources()
            hooks.validate_lock_files()
            hooks.list_host_zombie_processes()
        else:  # Xdist worker thread
            hooks.download_docker_images()
            hooks.init_ovms_config_retrieved_from_master(config)

        hooks.setup_nginx()

        # Let know that pytest was successfully configured
        config.configured = True


    def pytest_unconfigure(config):
        if getattr(config, "configured", None) is not True:
            # Check if pytest_configure() was done successfully, if not: logger would be in invalid state so disable.
            for _logger in logger.manager.loggerDict.values():
                _logger.disabled = True

        try:
            if is_xdist_master():
                hooks.remove_ports_reservation(config)
                hooks.cleanup_tmp_repos_dir(config)
                hooks.teardown_environment()
                if machine_is_reserved_for_test_session:
                    hooks.clear_lockfiles()
        except Exception as e:      # pylint: disable=broad-exception-caught
            error_msg = str(e)
            print(error_msg)
            sys.exit(error_msg)


    def pytest_configure_node(node):
        node.workerinput[TMP_REPOS_DIR_ARGUMENT] = node.config.tmp_repos_dir
        node.workerinput[CURRENT_TARGET_DEVICE_DICT_ARGUMENT] = node.config.current_target_device_dict


    MarksRegistry.MARK_ENUMS.extend([OvmsComponents])


    def pytest_sessionstart(session):
        hooks.get_session_start_info(session)


    @pytest.hookimpl(hookwrapper=True)
    def pytest_collection_modifyitems(session, config, items):
        """
        Support for running tests with component tags.
        Report all test component markers to mongo_reporter.
        """
        logger.info(f"Preparing tests for test session in the following folder: {session.startdir}")

        if pytest_keyword_filter:
            # Filter case insensitive
            deselected = [_item for _item in items if pytest_keyword_filter.lower() not in _item.name.lower()]
            if deselected:
                hooks.deselect_items(items, config, deselected)

        yield  # deselect items in default hook way via keyword ('-k')

        if config.option.collectonly:
            hooks.log_skip_statistic(items)

        deselected = hooks.preprocess_collected_items(items)
        if deselected:
            hooks.deselect_items(items, config, deselected)

        hooks.set_divide_target_device_per_worker(items)

        random.Random(7).shuffle(items)


    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(item: "Item"):
        """
         Perform the runtest protocol for a single test item.
         The default runtest protocol is this (see individual hooks for full details):
             pytest_runtest_logstart(nodeid, location)
             Setup phase:
                     call = pytest_runtest_setup(item) (wrapped in CallInfo(when="setup"))
                     report = pytest_runtest_makereport(item, call)
                     pytest_runtest_logreport(report)
                     pytest_exception_interact(call, report) if an interactive exception occurred
             Call phase, if the setup passed and the setuponly pytest option is not set:
                     call = pytest_runtest_call(item) (wrapped in CallInfo(when="call"))
                     report = pytest_runtest_makereport(item, call)
                     pytest_runtest_logreport(report)
                     pytest_exception_interact(call, report) if an interactive exception occurred
             Teardown phase:
                     call = pytest_runtest_teardown(item, nextitem) (wrapped in CallInfo(when="teardown"))
                     report = pytest_runtest_makereport(item, call)
                     pytest_runtest_logreport(report)
                     pytest_exception_interact(call, report) if an interactive exception occurred
             pytest_runtest_logfinish(nodeid, location)
         """
        __root_logger = get_logger(None)
        if not item.keywords.get("skip"):
            fh = OvmsFileHandler(item)
            __root_logger.addHandler(fh)
        yield
        if not item.keywords.get("skip"):
            fh.close()
            __root_logger.removeHandler(fh)


    def pytest_generate_tests(metafunc):
        hooks.parametrize_tests(metafunc)
