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

import requests

import tests.functional.utils.assertions as assertions_module
from tests.functional.utils.assertions import (
    BadRIPValue,
    DmesgBpFilterFail,
    DmesgError,
    GeneralProtectionFault,
    GPUHangError,
    OOMKillError,
    OvmsCrashed,
    SegfaultError,
    TrapDivideError,
)
from tests.functional.utils.core import get_children_from_module
from tests.functional.utils.logger import get_logger
from tests.functional.utils.process import Process
from tests.functional.utils.test_framework import generate_test_object_name
from ovms.config import artifacts_dir
from tests.functional.utils.log_monitor import LogMonitor

logger = get_logger(__name__)
MAX_DMESG_LINES = 1000

DMESG_UNEXPECTED_MESSAGES = set()


class DummyLogMonitor(LogMonitor):
    def _get_unexpected_messages_regex(self):
        return []

    def _get_unexpected_messages(self):
        pass

    def _refresh(self):
        pass

    def get_all_logs(self):
        return []


class DmesgLogMonitor(LogMonitor):

    def _get_unexpected_messages_regex(self):
        return [DmesgBpFilterFail.regex]

    def _get_unexpected_messages(self):
        messages = [
            BadRIPValue.msg,
            SegfaultError.msg,
            GPUHangError.msg,
            GeneralProtectionFault.msg,
            TrapDivideError.msg,
            OOMKillError.msg,
        ]
        if self.ovms_pid:
            messages.append(f"ovms[{self.ovms_pid}]")
        return messages

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._proc = Process()
        self._proc.set_log_silence()
        self._proc.disable_check_stderr()
        self._start_test_marker = ""
        self.ovms_pid = kwargs.get("ovms_pid", None)

    def _calculate_current_offset(self, start_log_marker):
        log_cnt = len(self._read_lines)
        if start_log_marker:
            while log_cnt > 0:
                start_log_entry = self._read_lines[log_cnt - 1]
                if start_log_entry == start_log_marker:
                    self.current_offset = log_cnt
                    break
                log_cnt -= 1

        if not self._start_test_marker and self._read_lines:
            assert len(self._read_lines) > self.current_offset
            self._start_test_marker = self._read_lines[self.current_offset - 1]

        return log_cnt

    def _refresh(self, start_position=None):
        if start_position is None:
            start_position = self._read_lines[self.current_offset - 1] if self._read_lines else ""
        # skip non-fatal log levels ("notice,info,debug")
        level = ",".join(["emerg", "alert", "crit", "err", "warn", "info"])
        stdout = self._proc.run_and_check(f"dmesg -T --level={level}", exception_type=DmesgError)
        self._read_lines = stdout.splitlines()
        self.current_offset = self._calculate_current_offset(start_position)

    def _generate_smoke_dmesg_log_name(self):
        name = generate_test_object_name(separator="_")
        filename = f"dmesg_smoke_{name}.log"
        return filename

    def get_all_logs(self):
        try:
            self._refresh(self._start_test_marker)
        except requests.exceptions.HTTPError as e:
            raise OvmsCrashed(msg=str(e), dmesg_log=self.get_logs_as_txt())
        return self._read_lines[self.current_offset:]

    def reset_to_logger_creation(self):
        if self._start_test_marker:
            self.current_offset = self._calculate_current_offset(self._start_test_marker)

    def filter_unexpected_messages(
        self, log_entry, found_unexpected_messages, unexpected_messages, unexpected_messages_re
    ):
        super().filter_unexpected_messages(
            log_entry, found_unexpected_messages, unexpected_messages, unexpected_messages_re
        )
        if log_entry.split("] ")[-1].startswith(" in lib"):
            logger.info("Get the next dmesg line to catch lib")
            found_unexpected_messages.append(log_entry)

    def raise_on_unexpected_messages(self, logs=None, filter_known_messages=False):
        unexpected_messages = self.check_for_unexpected_messages(logs=logs)
        DMESG_UNEXPECTED_MESSAGES.update(unexpected_messages)
        if filter_known_messages:
            unexpected_messages = [
                unexpected_message
                for unexpected_message in unexpected_messages
                if unexpected_message not in DMESG_UNEXPECTED_MESSAGES
            ]
        logger.info(f"Dmesg OVMS process ID: {self.ovms_pid}")
        if unexpected_messages:
            dmesg_exceptions = get_children_from_module(DmesgError, assertions_module)  # [(name, class_def), ...]
            for name, exception_class in dmesg_exceptions:
                msg = getattr(exception_class, "msg", None)
                regex = getattr(exception_class, "regex", None)
                if (msg and any(filter(lambda x: msg in x, unexpected_messages))) or (
                    regex and any(filter(lambda x: regex.match(x), unexpected_messages))
                ):
                    logger.error(f"Found unexpected message in dmesg logs: {msg}")
                    raise exception_class("\n".join(unexpected_messages), dmesg_log=logs)

    def dump_dmesg_logs_into_file(self, level=None):
        sysctl_stdout = self._proc.run_and_check("sysctl kernel.dmesg_restrict")
        if re.match(r"kernel.dmesg_restrict = (\d)", sysctl_stdout).group(1) != "0":
            message = (
                "Missing permissions to use dmesg without sudo. " "Please set 'sudo sysctl -w kernel.dmesg_restrict=0'."
            )
            raise DmesgError(message)
        filename = self._generate_smoke_dmesg_log_name()
        if level is None:
            level = ["emerg", "alert", "crit", "err", "warn"]
        level = ",".join(level)
        os.makedirs(artifacts_dir, exist_ok=True)
        file_path = os.path.join(artifacts_dir, filename)
        stdout = self._proc.run_and_check(
            f"dmesg --level={level} -T | tail -n {MAX_DMESG_LINES} | tee {file_path}", exception_type=DmesgError
        )
        logger.info(f"Logs saved {file_path}")
        return file_path, stdout

    def search_for_error_in_dmesg_file(self, stdout):
        if stdout is not None:
            logs = stdout.splitlines()
            self.raise_on_unexpected_messages(logs=logs)

    def clear_dmesg_buffer(self):
        self._proc.run_and_check("sudo dmesg -C")
        logger.info("Dmesg cleared")
