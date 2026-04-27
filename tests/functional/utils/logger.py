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

import base64
import hashlib
import inspect
import logging
import os
import re
import weakref
from datetime import datetime
from typing import Generator, List, Tuple, Union, cast

from tests.functional.constants.os_type import OsType, get_host_os
from tests.functional.utils.helpers import get_xdist_worker_count

from ovms import config


SEPARATOR = "=" * 20
FIXTURE_SEPARATOR = "*" * 20
UNDEFINED = "<undefined>"
UNDEFINED_BASE64 = base64.b64encode(UNDEFINED.encode('utf-8'))
API = "api"
LOCALHOST = "localhost"

ONE_K = 1024
ONE_M = ONE_K * ONE_K


log_username = f"- [{config.host_os_user}] " if config.log_username else ""
worker_count = get_xdist_worker_count()
worker_id = os.environ.get("PYTEST_XDIST_WORKER", "")
worker_string = f"[{worker_id}] " if worker_count > 0 else ""
logger_format = config.logger_format(worker_string, log_username)

logging.basicConfig(
    level=config.logging_level,
    format=logger_format,
)


class Chunks(Generator):
    """
    generator yielding tuple: no of part, number of parts, and part of the input list
    """
    def __init__(self, seq: List[str], max_number_of_elements: int = 1000) -> None:
        super().__init__()
        self.seq = tuple(seq)
        assert max_number_of_elements > 0, "Incorrect number of elements, should be more than zero"
        self.chunk_len = max_number_of_elements
        self.no_of_chunks = (len(self.seq) // self.chunk_len) + 1
        self.current_chunk = 0
        self.index_iterator = iter(range(0, len(self.seq), self.chunk_len))

    def __next__(self) -> Tuple[int, int, list]:
        return self.send(None)

    def __iter__(self) -> 'Chunks':
        return self

    def send(self, ignored_value) -> Tuple[int, int, list]:
        index = next(self.index_iterator)
        return_chunk = self.current_chunk, self.no_of_chunks, list(self.seq[index:index + self.chunk_len])
        self.current_chunk += 1
        return return_chunk

    def throw(self, typ, val=None, tb=None):
        raise StopIteration

    def close(self) -> None:
        raise GeneratorExit


class SensitiveKeysStrippingFilter(logging.Filter):
    instance = None
    sensitive_pairs = None  # type: dict
    sensitive_values_to_be_masked = None  # type: re

    def __new__(cls) -> 'SensitiveKeysStrippingFilter':
        if cls.instance is None:
            cls.instance = super().__new__(cls)
            cls.sensitive_pairs = cls.gather_sensitive_pairs()
            cls.sensitive_values_to_be_masked = list(cls.sensitive_pairs.values())
        return cls.instance

    @classmethod
    def build_sensitive_values_regexp(cls) -> re:
        return re.compile(
            "|".join([fr"{var}"
                      for var in cls.sensitive_pairs.values()]))

    @classmethod
    def gather_sensitive_pairs(cls) -> dict:
        return dict([(var, getattr(config, var, None))
                     for var in dir(config)
                     if cls.is_matching_variable(var)])

    @staticmethod
    def is_matching_variable(var) -> bool:
        if config.sensitive_keys_to_be_masked.match(var):
            var_value = getattr(config, var, UNDEFINED)
            if var_value is not UNDEFINED and \
                    isinstance(var_value, str) and \
                    len(var_value) > 0:
                return True
        return False

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.strip_sensitive_data(record.msg)
        record.args = self.filter_args(record.args)
        return True

    def filter_args(self, args: Union[dict, tuple]) -> Union[dict, tuple]:
        if not isinstance(args, (dict, tuple)):
            return args
        if isinstance(args, dict):
            args = self.strip_sensitive_data(args)
        else:
            args = tuple(self.strip_sensitive_data(arg) for arg in args)
        return args

    def strip_sensitive_data(self, data: Union[dict, str]) -> Union[dict, str]:
        if config.strip_sensitive_data:
            if isinstance(data, str) and len(data) > 0:
                data = self.strip_sensitive_str_values(data)
            elif isinstance(data, dict):
                data = self.strip_sensitive_dict_values(data.copy())
        return data

    def strip_sensitive_dict_values(self, data: dict) -> dict:
        for key, value in data.items():
            if value in self.sensitive_values_to_be_masked:
                data[key] = "***<masked by logger>***"
        return data

    def strip_sensitive_str_values(self, data: str) -> str:
        stripped_data = data
        for sensitive_value_to_be_masked in self.sensitive_values_to_be_masked:
            stripped_data = stripped_data.replace(sensitive_value_to_be_masked, "***<masked by logger>***")
        return stripped_data


class LoggerType(object):
    """Logger types definitions"""
    HTTP_REQUEST = "http_request"
    HTTP_RESPONSE = "http_response"
    REMOTE_LOGGER = "remote logger"
    SHELL_COMMAND = "shell command"
    STEP_LOGGER = "STEP"
    FIXTURE_LOGGER = "FIXTURE"
    FINALIZER_LOGGER = "FINALIZER"


class Logger(logging.Logger):
    """src: https://stackoverflow.com/a/22586200"""
    MIN_NUMBER_OF_LINES_TO_PRESENT_FINAL_MSG = 20
    VERBOSE = 5

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        # noinspection PyTypeChecker
        self.last_record = None  # type: weakref.ReferenceType
        logging.addLevelName(self.VERBOSE, "VERBOSE")

    # pylint: disable=too-many-arguments
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo: Union[None, bool] = None):
        record = super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)
        self.last_record = weakref.ref(record)  # type: weakref.ReferenceType
        return record

    # pylint: disable=too-many-arguments
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False,
             list_of_strings: List[str] = None,
             chunk_len: int = 1000,
             chunk_msg: str = None,
             final_msg: str = None):
        try:
            super()._log(level, msg, args, exc_info, extra, stack_info)
        except Exception as exc:
            print(str(exc))
            print(msg)
        self.log_list_of_strings(level, chunk_msg, args, exc_info, extra,
                                 stack_info, list_of_strings, chunk_len, final_msg)

    def findCaller(self, stack_info: bool = False, stacklevel: int = 1):
        last_record = self.last_record() if self.last_record is not None else None  # type: logging.LogRecord
        if last_record is not None:
            return last_record.pathname, last_record.lineno, last_record.funcName, last_record.stack_info
        return super().findCaller(stack_info=stack_info)

    # pylint: disable=too-many-arguments
    def log_list_of_strings(self, level, chunk_msg, args, exc_info=None, extra=None, stack_info=False,
                            list_of_strings: List[str] = None,
                            chunk_len: int = 1000,
                            final_msg: str = None):
        fn, lno, func, sinfo = self.findCaller(stack_info=stack_info)
        if list_of_strings is not None and len(list_of_strings):
            chunks = Chunks(list_of_strings, max_number_of_elements=chunk_len)
            if chunks.no_of_chunks > 1:
                chunk_msg = chunk_msg.rstrip() if chunk_msg is not None else "Presenting chunk"
                chunk_msg = " ".join([chunk_msg.rstrip(), "({index}/{no_of_chunks}):\n{chunk}\n"])
            else:
                chunk_msg = "\n{chunk}\n"
            list_chunk = []
            for chunk_number, no_of_chunks, list_chunk in chunks:
                formatted_chunk_msg = chunk_msg.format(index=chunk_number,
                                                       no_of_chunks=no_of_chunks,
                                                       chunk="\n".join(list_chunk))
                chunk_record = self.makeRecord(self.name, level, fn, lno, formatted_chunk_msg, args,
                                               exc_info, func, extra, sinfo)
                self.handle(chunk_record)
            else:
                if len(list_chunk) > self.MIN_NUMBER_OF_LINES_TO_PRESENT_FINAL_MSG:
                    final_msg = final_msg.rstrip() if final_msg is not None else "End of presenting chunks"
                    if chunks.no_of_chunks > 1:
                        final_msg = " ".join([final_msg, f"Presented {chunks.no_of_chunks} chunks."])
                    final_record = self.makeRecord(self.name, level, fn, lno, final_msg, args,
                                                   exc_info, func, extra, sinfo)
                    self.handle(final_record)

    def verbose(self, msg, *args, **kwargs):
        if self.isEnabledFor(self.VERBOSE):
            self._log(self.VERBOSE, msg, args, **kwargs)


logging.setLoggerClass(Logger)
logging.addLevelName(Logger.VERBOSE, "VERBOSE")

__LOGGING_LEVEL = config.logging_level


def get_logger(name) -> Logger:
    logger = logging.getLogger(name)
    logger.addFilter(SensitiveKeysStrippingFilter())
    logger.setLevel(__LOGGING_LEVEL)
    return cast(Logger, logger)


def step(message):
    caller = inspect.stack()[1][3]
    _log_separator(logger_type=LoggerType.STEP_LOGGER, separator=SEPARATOR, caller=caller, message=message)


def log_fixture(message, separator=FIXTURE_SEPARATOR):
    caller = inspect.stack()[1][3]
    _log_separator(logger_type=LoggerType.FIXTURE_LOGGER, separator=separator, caller=caller, message=message)


def _log_separator(logger_type, separator, caller, message):
    get_logger(logger_type).info(f"{separator} {caller}: {message} {separator}")


def line_trimmer(line: str, max_number_of_elements: int = 1 * ONE_K // 4):
    if len(line) > max_number_of_elements:
        line = "t: " + \
               line[:max_number_of_elements // 2] + \
               "[...]" + \
               line[-max_number_of_elements // 2:]
    return line


def list_trimmer(seq: list, max_number_of_elements: int = 4 * ONE_K):
    if len(seq) > max_number_of_elements:
        first_element = [f"Too long output was trimmed! Original len {len(seq)}, "
                         f"showing first and last {max_number_of_elements // 2} lines:"]
        seq = first_element + seq[:max_number_of_elements // 2] + ["", "[...]", ""] + seq[-max_number_of_elements // 2:]
    return seq


def log_trimmer(logs: str):
    logs_list = logs.split(sep="\n")
    logs_list = [line_trimmer(line) for line in logs_list]
    logs_trimmed = list_trimmer(seq=logs_list, max_number_of_elements=ONE_K)
    logs = "\n".join(logs_trimmed)
    return logs


def sanitize_node(name_or_node_id):
    name_or_node_id = "__".join(name_or_node_id.split("/"))
    name_or_node_id = "..".join(name_or_node_id.split("::"))
    name_or_node_id = "-".join(name_or_node_id.split(" "))
    return name_or_node_id


class OvmsFileHandler(logging.FileHandler):
    def __init__(self, item: Union["Item", str] = None,
                 mode='a', encoding=None, delay=False):
        self.filename = self.log_file_name(item)
        file_path = os.path.join(config.test_log_directory, self.filename)
        if item is not None:
            item._log_file_name = self.filename
        super().__init__(file_path, mode, encoding, delay)
        fmt = logging.Formatter(logger_format)
        self.setFormatter(fmt)

    @staticmethod
    def safe_node_id(item: "Item"):
        return sanitize_node(item.nodeid)

    @classmethod
    def log_file_name(cls, item: "Item" = None):
        if isinstance(item, str):
            file_name = item
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            worker = f"_{worker_id}" if worker_count > 0 else ""
            prefix = f"{cls.safe_node_id(item)}" if item is not None else "api_tests"
            file_name = f"{prefix}{worker}_{timestamp}.log"
        if len(file_name.encode('utf-8')) > 255:
            file_name = cls.generate_short_log_file_name(file_name)
        if get_host_os() == OsType.Windows:
            # replace characters not supported by Windows: \/:*?"<>|
            file_name = re.sub(r'[\\\/\:\*\?\\"\<\>\|]', "_", file_name)
        return file_name

    @classmethod
    def generate_short_log_file_name(cls, file_name):
        encoded_str = file_name.encode()
        hash_obj = hashlib.sha1(encoded_str)
        hexa_value = hash_obj.hexdigest()
        return f"{file_name.split('.log')[0][:246]}_{hexa_value[:4]}.log"
