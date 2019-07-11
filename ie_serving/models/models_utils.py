class ErrorCode:
    OK = 0
    CANCELLED = 1
    UNKNOWN = 2
    INVALID_ARGUMENT = 3
    DEADLINE_EXCEEDED = 4
    NOT_FOUND = 5
    ALREADY_EXISTS = 6
    PERMISSION_DENIED = 7
    UNAUTHENTICATED = 16
    RESOURCE_EXHAUSTED = 8
    FAILED_PRECONDITION = 9
    ABORTED = 10
    OUT_OF_RANGE = 11
    UNIMPLEMENTED = 12
    INTERNAL = 13
    UNAVAILABLE = 14
    DATA_LOSS = 15
    DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_ \
        = 20


class ModelVersionState:
    UNKNOWN = 0
    START = 10
    LOADING = 20
    AVAILABLE = 30
    UNLOADING = 40
    END = 50


_ERROR_MESSAGE = {
    ModelVersionState.START: {
        ErrorCode.OK: "",   # "Version detected"
    },
    ModelVersionState.LOADING: {
        ErrorCode.OK: "",   # "Version is being loaded",
        ErrorCode.UNKNOWN: "Error occurred while loading version"
    },
    ModelVersionState.AVAILABLE: {
        ErrorCode.OK: "",   # "Version available"
    },
    ModelVersionState.UNLOADING: {
        ErrorCode.OK: "",   # "Version is scheduled to be deleted"
    },
    ModelVersionState.END: {
        ErrorCode.OK: "",   # "Version has been removed"
    },
}

_STATE_NAME = {
    ModelVersionState.UNKNOWN: "UNKNOWN",
    ModelVersionState.START: "START",
    ModelVersionState.LOADING: "LOADING",
    ModelVersionState.AVAILABLE: "AVAILABLE",
    ModelVersionState.UNLOADING: "UNLOADING",
    ModelVersionState.END: "END",
}

_ERROR_CODE_NAME = {
    ErrorCode.OK: "OK",
    ErrorCode.CANCELLED: "CANCELLED",
    ErrorCode.UNKNOWN: "UNKNOWN",
    ErrorCode.INVALID_ARGUMENT: "INVALID_ARGUMENT",
    ErrorCode.DEADLINE_EXCEEDED: "DEADLINE_EXCEEDED",
    ErrorCode.NOT_FOUND: "NOT_FOUND",
    ErrorCode.ALREADY_EXISTS: "ALREADY_EXISTS",
    ErrorCode.PERMISSION_DENIED: "PERMISSION_DENIED",
    ErrorCode.UNAUTHENTICATED: "UNAUTHENTICATED",
    ErrorCode.RESOURCE_EXHAUSTED: "RESOURCE_EXHAUSTED",
    ErrorCode.FAILED_PRECONDITION: "FAILED_PRECONDITION",
    ErrorCode.ABORTED: "ABORTED",
    ErrorCode.OUT_OF_RANGE: "OUT_OF_RANGE",
    ErrorCode.UNIMPLEMENTED: "UNIMPLEMENTED",
    ErrorCode.INTERNAL: "INTERNAL",
    ErrorCode.UNAVAILABLE: "UNAVAILABLE",
    ErrorCode.DATA_LOSS: "DATA_LOSS",
    ErrorCode
        .DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_: # noqa
        "DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_" # noqa
}


class ModelVersionStatus:
    def __init__(self, version: int):
        self.version = version
        self.state = ModelVersionState.START
        self.status = {"error_code": ErrorCode.OK,
                       "error_message": _ERROR_MESSAGE[
                           ModelVersionState.START][ErrorCode.OK]}

    def set_loading(self, error_code=ErrorCode.OK):
        self.state = ModelVersionState.LOADING
        self.status["error_code"] = error_code
        self.status["error_message"] = _ERROR_MESSAGE[
            ModelVersionState.LOADING][error_code]

    def set_available(self, error_code=ErrorCode.OK):
        self.state = ModelVersionState.AVAILABLE
        self.status["error_code"] = error_code
        self.status["error_message"] = _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][error_code]

    def set_unloading(self, error_code=ErrorCode.OK):
        self.state = ModelVersionState.UNLOADING
        self.status["error_code"] = error_code
        self.status["error_message"] = _ERROR_MESSAGE[
            ModelVersionState.UNLOADING][error_code]

    def set_end(self, error_code=ErrorCode.OK):
        self.state = ModelVersionState.END
        self.status["error_code"] = error_code
        self.status["error_message"] = _ERROR_MESSAGE[
            ModelVersionState.END][error_code]

    def prepare_response(self):
        return {
            "version": self.version,
            "state": _STATE_NAME[self.state],
            "status": {
                "error_code": _ERROR_CODE_NAME[self.status['error_code']],
                "error_message": self.status['error_message']
            }
        }


def prepare_statuses(statuses: list):
    result = []
    [result.append(status.prepare_response()) for status in statuses]
    return result
