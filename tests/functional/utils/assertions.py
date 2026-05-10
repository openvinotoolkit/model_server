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

import grpc
import json
import os
import pytest
import re
import yaml
from pathlib import Path
from typing import Callable, Type

from tests.functional.utils.logger import get_logger
from tests.functional.constants.ovms import CurrentOvmsType
from tests.functional.constants.paths import Paths

logger = get_logger(__name__)
CPP_STD_EXCEPTION = "std::exception"


# pylint: disable=too-many-instance-attributes
class OvmsTestException(AssertionError):
    def __init__(self, msg=None, ovms_log=None, dmesg_log=None, context=None, **kwargs):
        super().__init__(msg)
        self.ovms_log = ovms_log
        self.dmesg_log = dmesg_log
        self.ovms_type = CurrentOvmsType.ovms_type
        self.context = context

    def set_process_details(self, cmd=None, retcode=None, stdout=None, stderr=None):
        self.cmd = cmd
        self.retcode = retcode
        self.stdout = stdout
        self.stderr = stderr

    def get_process_details(self):
        cmd = getattr(self, "cmd", "")
        retcode = getattr(self, "retcode", "")
        stdout = getattr(self, "stdout", "")
        stderr = getattr(self, "stderr", "")
        return cmd, retcode, stdout, stderr

    def __str__(self):
        msg = super().__str__()
        header = f"{self.ovms_type}"
        msg = f"\n{header}\n{msg}"
        return msg


class AggregatedMultipleOvmsTestExceptions(OvmsTestException):
    def __init__(self, multiple_exceptions):
        self.multiple_ovms_exceptions = multiple_exceptions

    def __str__(self):
        return "\n".join(str(e) for e in self.multiple_ovms_exceptions)


class UnexpectedResponseError(OvmsTestException):
    def __init__(self, status=None, error_message=None, message=None):
        message = message or f"Code:{status} Message:{error_message}"
        super(UnexpectedResponseError, self).__init__(message)
        self.status = status
        self.error_message = error_message


class TemplateMessageException(Exception):
    TEMPLATE = ""

    def __init__(self, message=None):
        super().__init__(self.TEMPLATE.format(message))


def assert_raises_exception(exception: Type[BaseException], output, callable_obj, *args, **kwargs):
    with pytest.raises(exception) as e:
        callable_obj(*args, **kwargs)
    assert output in str(e.value), \
        f"Expected output:\n{output}\nnot found in exception {exception.__class__.__name__} value:\n{str(e.value)}"


def assert_raises_exception_with_pattern(exception: Type[BaseException], pattern, callable_obj, *args, **kwargs):
    with pytest.raises(exception) as e:
        callable_obj(*args, **kwargs)
    assert pattern.search(str(e.value)), \
        f"Expected output:\n{pattern}\nnot found in exception {exception.__class__.__name__} value:\n{str(e.value)}"


def get_mediapipe_details_from_context(context):
    client_input_data, client_output_data = None, None
    ovms_session = context.ovms_sessions[0]
    log_monitor = ovms_session.ovms.create_log(True)
    ovms_log = log_monitor.get_logs_as_txt()
    config_file = os.path.join(ovms_session.ovms.container_folder, Paths.MODELS_PATH_NAME, Paths.CONFIG_FILE_NAME)
    config = json.loads(Path(config_file).read_text())
    mediapipe_model = [model for model in ovms_session.models if model.is_mediapipe][0]
    src_code = [calc.src_file_path for calc in mediapipe_model.calculators]
    graphs = mediapipe_model.graphs
    request = getattr(context, "request", None)
    if request:
        client_input_data = request.inputs
        client_output_data = request.outputs
    return ovms_log, config, graphs, src_code, client_input_data, client_output_data


def _assert_status_code_and_message(status, error_message_phrase, status_code, error_msg, e, context=None):
    try:
        error_msg = yaml.load(error_msg, Loader=yaml.Loader)  # convert dict saved as string
    except (yaml.scanner.ScannerError, yaml.parser.ParserError) as exception:
        e = exception
        pass
    error_msg = error_msg["error"] if getattr(error_msg, "error", None) is not None else str(error_msg)
    assert error_message_phrase in error_msg, \
        f"Expected output:\n{error_message_phrase}\nnot found in exception {e.__class__.__name__} value:\n{error_msg}"
    assert status == status_code, f"Not expected status code found: got: {status_code}, expected: {status}"


def assert_raises_http_exception(
        status: int, error_message_phrase: str, callable_obj: Callable, context=None, *args, **kwargs
):
    with pytest.raises(UnexpectedResponseError) as e:
        callable_obj(*args, **kwargs)
    _assert_status_code_and_message(status, error_message_phrase,
                                    e.value.status, e.value.error_message, e, context, *args)


def assert_raises_grpc_exception(
        status, error_message_phrase: str, callable_obj: Callable, context=None, *args, **kwargs
):
    with pytest.raises(grpc.RpcError) as e:
        callable_obj(*args, **kwargs)
    _assert_status_code_and_message(status, error_message_phrase,
                                    e.value._state.code.value[0], e.value._state.details, e, context, *args)


class InstallPkgVersionException(OvmsTestException):
    pass


class AptInstallException(OvmsTestException):
    pass


class UpgradePkgException(OvmsTestException):
    pass


class InvalidMetadataException(OvmsTestException):
    pass


class ModelNotReadyException(OvmsTestException):
    pass


class ServerNotLiveException(OvmsTestException):
    pass


class ServerNotReadyException(OvmsTestException):
    pass


class SdlException(OvmsTestException):
    pass


class StatefulModelGeneralException(OvmsTestException):
    pass


class ModelCacheFailure(OvmsTestException):
    pass


class ModelCacheIncorrectNumberOfCacheFiles(ModelCacheFailure):
    def __int__(self, msg=None):
        if msg is None:
            msg = "Incorrect number of expected cache files"
        super().__init__(msg)


class PodNotReadyException(OvmsTestException):
    pass


class PodCreationException(OvmsTestException):
    pass


class LogMessageNotFoundException(OvmsTestException):
    pass


class KubeCtlApplyException(OvmsTestException):
    pass


class AutomaticCodeReviewException(OvmsTestException):
    pass


class StartOvmsException(OvmsTestException):
    pass


class ExampleClientsError(OvmsTestException):
    pass


class CurlArtifactsError(OvmsTestException):
    pass


class UnzipError(OvmsTestException):
    pass


class CreateVenvError(OvmsTestException):
    pass


class PipInstallError(OvmsTestException):
    pass


class BuildCustomNodeError(OvmsTestException):
    pass


class DockerBuildError(OvmsTestException):
    pass


class OvmsCrashed(OvmsTestException):
    pass


class DmesgError(OvmsTestException):
    msg = "dmesg error"


class DockerCannotCloseProperly(OvmsTestException):
    pass


class DmesgBpFilterFail(DmesgError):
    regex = re.compile("bpfilter: .+ fail .+")


class BadRIPValue(DmesgError):
    msg = "Bad RIP value"


class SegfaultError(DmesgError):
    msg = "segfault"


class GPUHangError(DmesgError):
    msg = "GPU HANG"


class GeneralProtectionFault(DmesgError):
    msg = "general protection fault"


class TrapDivideError(DmesgError):
    msg = "trap divide error"


class OOMKillError(DmesgError):
    msg = "oom-kill"


class DownloadError(OvmsTestException):
    pass


class UnwantedMessageError(OvmsTestException):
    pass


class DocumentationError(OvmsTestException):
    pass


class MissingLinks(OvmsTestException):
    pass


class SpellingError(OvmsTestException):
    pass


class NginxException(OvmsTestException):
    phrase_re = re.compile(r"nginx: \[emerg\]")


class DocumentationCommandsException(OvmsTestException):
    pass


class InvalidReturnCodeException(OvmsTestException):
    pass


class ProvisioningFailure(OvmsTestException):
    pass


class CloudException(OvmsTestException):
    pass


class HTTPError(OvmsTestException):
    pass


class GitCloneException(OvmsTestException):
    pass


class GitBranchException(OvmsTestException):
    pass


class FuzzingError(OvmsTestException):
    pass


class NotSupported(OvmsTestException):
    pass


class AccuracyException(OvmsTestException):
    pass


class StreamingApiException(OvmsTestException):
    pass


class ValgrindException(OvmsTestException):
    pass


class ModelAnalyzerException(OvmsTestException):
    pass


class CapiException(OvmsTestException):
    pass


class CommonDockerException(OvmsTestException):
    pass


class ConvertModelException(OvmsTestException):
    pass


class OVVPException(OvmsTestException):
    pass


def get_exception_by_ovms_log(ovms_log_lines):
    exceptions_to_recognize = [NginxException]

    for line in ovms_log_lines:
        for e in exceptions_to_recognize:
            m = e.phrase_re.search(line)
            if m:
                return e, line
    return None
