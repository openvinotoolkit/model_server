class ErrorCode:
    OK = 0
    # CANCELLED = 1
    UNKNOWN = 2
    # INVALID_ARGUMENT = 3
    # DEADLINE_EXCEEDED = 4
    # NOT_FOUND = 5
    # ALREADY_EXISTS = 6
    # PERMISSION_DENIED = 7
    # UNAUTHENTICATED = 16
    # RESOURCE_EXHAUSTED = 8
    # FAILED_PRECONDITION = 9
    # ABORTED = 10
    # OUT_OF_RANGE = 11
    # UNIMPLEMENTED = 12
    # INTERNAL = 13
    # UNAVAILABLE = 14
    # DATA_LOSS = 15
    # DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_ \
    #    = 20


class ModelVersionState:
    # UNKNOWN = 0
    START = 10
    LOADING = 20
    AVAILABLE = 30
    UNLOADING = 40
    END = 50


_ERROR_MESSAGE = {
    ModelVersionState.START: {
        ErrorCode.OK: "",  # "Version detected"
    },
    ModelVersionState.LOADING: {
        ErrorCode.OK: "",  # "Version is being loaded",
        ErrorCode.UNKNOWN: "Error occurred while loading version"
    },
    ModelVersionState.AVAILABLE: {
        ErrorCode.OK: "",  # "Version available"
    },
    ModelVersionState.UNLOADING: {
        ErrorCode.OK: "",  # "Version is scheduled to be deleted"
    },
    ModelVersionState.END: {
        ErrorCode.OK: "",  # "Version has been removed"
    },
}

_STATE_NAME = {
    # ModelVersionState.UNKNOWN: "UNKNOWN",
    ModelVersionState.START: "START",
    ModelVersionState.LOADING: "LOADING",
    ModelVersionState.AVAILABLE: "AVAILABLE",
    ModelVersionState.UNLOADING: "UNLOADING",
    ModelVersionState.END: "END",
}

_ERROR_CODE_NAME = {
    ErrorCode.OK: "OK",
    # ErrorCode.CANCELLED: "CANCELLED",
    ErrorCode.UNKNOWN: "UNKNOWN",
    # ErrorCode.INVALID_ARGUMENT: "INVALID_ARGUMENT",
    # ErrorCode.DEADLINE_EXCEEDED: "DEADLINE_EXCEEDED",
    # ErrorCode.NOT_FOUND: "NOT_FOUND",
    # ErrorCode.ALREADY_EXISTS: "ALREADY_EXISTS",
    # ErrorCode.PERMISSION_DENIED: "PERMISSION_DENIED",
    # ErrorCode.UNAUTHENTICATED: "UNAUTHENTICATED",
    # ErrorCode.RESOURCE_EXHAUSTED: "RESOURCE_EXHAUSTED",
    # ErrorCode.FAILED_PRECONDITION: "FAILED_PRECONDITION",
    # ErrorCode.ABORTED: "ABORTED",
    # ErrorCode.OUT_OF_RANGE: "OUT_OF_RANGE",
    # ErrorCode.UNIMPLEMENTED: "UNIMPLEMENTED",
    # ErrorCode.INTERNAL: "INTERNAL",
    # ErrorCode.UNAVAILABLE: "UNAVAILABLE",
    # ErrorCode.DATA_LOSS: "DATA_LOSS",
    # ErrorCode.DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_:
    #    "DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_"
}
