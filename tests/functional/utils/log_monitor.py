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
from abc import ABC, abstractmethod
from datetime import datetime
from time import sleep

from tests.functional.utils.assertions import LogMessageNotFoundException, OvmsCrashed, UnwantedMessageError
from tests.functional.utils.logger import get_logger
from ovms.config import artifacts_dir, test_build_log_url, wait_for_messages_timeout

logger = get_logger(__name__)


class LogMonitor(ABC):

    def __init__(self, **kwargs):
        self.context = None
        self._read_lines = []
        self.current_offset = 0
        self.logger_creation_start_offset = 0

    def _read_log_line(self):
        log_line = None
        if self.current_offset >= len(self._read_lines):
            self._refresh()

        if self.current_offset < len(self._read_lines):
            log_line = self._read_lines[self.current_offset]
            self.current_offset += 1
        return log_line

    def _refresh(self, start_position=None):
        return self.get_all_logs()

    @abstractmethod
    def get_all_logs(self):
        raise NotImplemented()

    def get_logs_as_txt(self):
        return "\n".join(self.get_all_logs())

    @abstractmethod
    def _get_unexpected_messages(self):
        return []

    @abstractmethod
    def _get_unexpected_messages_regex(self):
        return []

    def raise_on_unexpected_messages(self, **kwargs):
        unexpected_messages = self.check_for_unexpected_messages()
        if not unexpected_messages:
            return

    def filter_unexpected_messages(
        self, log_entry, found_unexpected_messages, unexpected_messages, unexpected_messages_re
    ):
        if any(filter(lambda x: x in log_entry, unexpected_messages)) or any(
            filter(lambda x: x.match(log_entry), unexpected_messages_re)
        ):
            found_unexpected_messages.append(log_entry)

    def check_for_unexpected_messages(self, logs=None):
        if logs is None:
            logs = self.get_all_logs()

        unexpected_messages = self._get_unexpected_messages()
        unexpected_messages_re = self._get_unexpected_messages_regex()

        found_unexpected_messages = []
        for log_entry in logs:
            self.filter_unexpected_messages(
                log_entry, found_unexpected_messages, unexpected_messages, unexpected_messages_re
            )
        return found_unexpected_messages

    def cleanup(self, filename):
        logs = self.get_all_logs()
        found_unexpected_messages = self.check_for_unexpected_messages(logs)
        path = self.save_to_file(filename, logs)
        return path, found_unexpected_messages

    @staticmethod
    def save_to_file(filename, logs):
        os.makedirs(artifacts_dir, exist_ok=True)
        file_path = os.path.join(artifacts_dir, filename)
        with open(file_path, "w", encoding="utf-8") as fd:
            for line in logs:
                fd.write(f"{line}\n")
        logger.info(f"Logs saved: {os.path.join(test_build_log_url, 'artifact', 'test_log', filename)}")
        return file_path

    def reset_to_logger_creation(self):
        self.current_offset = self.logger_creation_start_offset

    def flush(self):
        self.current_offset = len(self._read_lines)

    def wait_for_messages(
        self,
        messages_to_find,
        break_msg_list=None,
        raise_exception_if_not_found=True,
        timeout=None,
        callbacks=[],
        ovms_instance=None,
        check_ovms_running=True,
        all_messages=False,
    ):
        break_msg_list = [] if break_msg_list is None else break_msg_list
        if timeout is None:
            timeout = wait_for_messages_timeout

        messages_found = False
        found_lines = []
        if messages_to_find:
            if isinstance(messages_to_find, str):
                messages_to_find = [messages_to_find]

            messages_to_find_vs_results_map = dict.fromkeys(messages_to_find, None)
            start = datetime.now()
            while not messages_found and (datetime.now() - start).total_seconds() <= timeout:
                log_line = self._read_log_line()
                if log_line is None:
                    if not check_ovms_running:
                        continue
                    else:
                        if not self.is_ovms_running():
                            break  # OVMS is not running, no new output is expected.
                    # Run callbacks at idle.
                    for callback in callbacks:
                        callback()
                    sleep(1)
                    continue

                found_lines.append(log_line)

                for specific_msg in messages_to_find_vs_results_map:
                    if messages_to_find_vs_results_map[specific_msg] is None:
                        if isinstance(specific_msg, str):
                            messages_to_find_vs_results_map[specific_msg] = log_line if specific_msg in log_line \
                                else None
                        elif isinstance(specific_msg, tuple):
                            for msg in specific_msg:
                                if msg in log_line:
                                    messages_to_find_vs_results_map[specific_msg] = log_line
                                    break
                        else:
                            raise NotImplementedError()

                if any(map(lambda break_msg: break_msg in log_line, break_msg_list)):
                    ovms_log = self.get_logs_as_txt()
                    logger.info(ovms_log)
                    raise UnwantedMessageError(
                        f"Found message: '{log_line}'", ovms_log=ovms_log, context=self.context
                    )

                if all_messages:
                    messages_found = all(x for x in messages_to_find_vs_results_map.values())
                else:
                    messages_found = any(x for x in messages_to_find_vs_results_map.values())
            self._log_search_info(
                raise_exception_if_not_found,
                messages_found,
                found_lines,
                messages_to_find_vs_results_map,
                ovms_instance,
            )

        # Run all callbacks at end.
        # Callbacks can be used to validate proper ovms load at many sources (ie: dmesg)
        for callback in callbacks:
            callback()
        return messages_found

    def find_messages(self, messages_to_find, raise_exception_if_not_found=False):
        all_messages_found = False
        found_lines = []

        if messages_to_find:
            if messages_to_find is str:
                messages_to_find = [messages_to_find]

            messages_to_find_vs_results_map = dict.fromkeys(messages_to_find, None)
            while not all_messages_found:
                log_line = self._read_log_line()
                if log_line is None:
                    break
                else:
                    found_lines.append(log_line)

                for specific_msg in messages_to_find_vs_results_map:
                    if messages_to_find_vs_results_map[specific_msg] is None:
                        messages_to_find_vs_results_map[specific_msg] = log_line if specific_msg in log_line else None

                all_messages_found = all([x for x in messages_to_find_vs_results_map.values()])
            self._log_search_info(
                raise_exception_if_not_found, all_messages_found, found_lines, messages_to_find_vs_results_map
            )
        return all_messages_found, messages_to_find_vs_results_map

    def _log_search_info(
        self,
        raise_exception_if_not_found,
        all_messages_found,
        found_lines,
        messages_to_find_vs_results_map,
        ovms_instance=None,
    ):
        if all_messages_found:
            for msg, log_entry in messages_to_find_vs_results_map.items():
                logger.debug(f"Find log procedure result: \n\tMessage: {msg} \n\tLog entry: {log_entry}")
        else:
            if raise_exception_if_not_found:
                missing_messages = [k for k, v in messages_to_find_vs_results_map.items() if v is None]
                found_lines_str = "\n".join(found_lines)
                error_msg = (
                    f"Unable to find following messages in logs: {missing_messages}\n= Ovms logs {18 * '='}\n"
                    f"{found_lines_str}\n{30 * '='}"
                )
                if len(missing_messages) != 0:
                    exception = OvmsCrashed if not self.is_ovms_running() else LogMessageNotFoundException
                    dmesg_log = None
                    if ovms_instance is not None:
                        ovms_instance._dmesg_log.raise_on_unexpected_messages()
                        dmesg_log = ovms_instance._dmesg_log.get_logs_as_txt()
                    raise exception(error_msg, ovms_log=self.get_logs_as_txt(), dmesg_log=dmesg_log)

    def is_ovms_running(self):
        return True
