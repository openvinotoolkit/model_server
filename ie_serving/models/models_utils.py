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
    ModelVersionState.AVAILABLE: {
        ErrorCode.OK: "Version available"
    },
    ModelVersionState.START: {
        ErrorCode.OK: "Version detected"
    },
    ModelVersionState.LOADING: {
        ErrorCode.OK: "Version is being loaded",
        ErrorCode.UNKNOWN: "Error occurred while loading version"
    },
    ModelVersionState.UNLOADING: {
        ErrorCode.OK: "Version is scheduled to be deleted"
    },
    ModelVersionState.END: {
        ErrorCode.OK: "Version has been removed"
    },
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

    def to_dict(self):
        return {
            "version": self.version,
            "state": self.state,
            "status": self.status
        }


def statuses_as_dicts(statuses: list):
    result = []
    [result.append(status.to_dict()) for status in statuses]
    return result
