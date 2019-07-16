import copy

from ie_serving.models.models_utils import ModelVersionState, ErrorCode, \
    _ERROR_MESSAGE, _STATE_NAME, _ERROR_CODE_NAME
from ie_serving.logger import get_logger

logger = get_logger(__name__)


class ModelVersionStatus:
    def __init__(self, model_name: str, version: int):
        self.model_name = model_name
        self.version = version
        self.state = ModelVersionState.START
        self.status = {"error_code": ErrorCode.OK,
                       "error_message": _ERROR_MESSAGE[
                           ModelVersionState.START][ErrorCode.OK]}
        self.log_status()

    def set_loading(self, error_code=ErrorCode.OK):
        self.state = ModelVersionState.LOADING
        self.status["error_code"] = error_code
        self.status["error_message"] = _ERROR_MESSAGE[
            ModelVersionState.LOADING][error_code]
        self.log_status()

    def set_available(self, error_code=ErrorCode.OK):
        self.state = ModelVersionState.AVAILABLE
        self.status["error_code"] = error_code
        self.status["error_message"] = _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][error_code]
        self.log_status()

    def set_unloading(self, error_code=ErrorCode.OK):
        self.state = ModelVersionState.UNLOADING
        self.status["error_code"] = error_code
        self.status["error_message"] = _ERROR_MESSAGE[
            ModelVersionState.UNLOADING][error_code]
        self.log_status()

    def set_end(self, error_code=ErrorCode.OK):
        self.state = ModelVersionState.END
        self.status["error_code"] = error_code
        self.status["error_message"] = _ERROR_MESSAGE[
            ModelVersionState.END][error_code]
        self.log_status()

    def log_status(self):
        state = _STATE_NAME[self.state]
        status = copy.deepcopy(self.status)
        status['error_code'] = _ERROR_CODE_NAME[status['error_code']]
        log_msg = {"state": state, "status": status}
        logger.debug("STATUS CHANGE: Version {} of model {} status change. "
                     "New status: {}".format(self.version, self.model_name,
                                             log_msg))
