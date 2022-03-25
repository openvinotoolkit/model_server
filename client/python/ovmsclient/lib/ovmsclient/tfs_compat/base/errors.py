#
# Copyright (c) 2021 Intel Corporation
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

from grpc import StatusCode
from http import HTTPStatus
from requests import ReadTimeout


class ModelServerError(Exception):
    pass


class ModelNotFoundError(ModelServerError):
    pass


class InvalidInputError(ModelServerError):
    pass


class BadResponseError(ModelServerError):
    pass


GRPC_ERROR_CODE_TO_EXCEPTION = {
    # If "message" key is present, then the source error message is overrode
    StatusCode.NOT_FOUND: {"class": ModelNotFoundError},
    StatusCode.INVALID_ARGUMENT: {"class": InvalidInputError},
    StatusCode.DEADLINE_EXCEEDED: {
                                      "class": TimeoutError,
                                      "message": "Request handling exceeded timeout"
                                  }
}

HTTP_ERROR_TYPE_TO_EXCEPTION = {
    ReadTimeout: {"class": TimeoutError, "message": "Request handling exceeded timeout"}
}

HTTP_ERROR_CODE_TO_EXCEPTION = {
    HTTPStatus.NOT_FOUND: {"class": ModelNotFoundError},
    HTTPStatus.BAD_REQUEST: {"class": InvalidInputError},
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE: {"class": ConnectionError},
    HTTPStatus.SERVICE_UNAVAILABLE: {"class": ConnectionError},
}


def raise_from_grpc(grpc_error):
    error = GRPC_ERROR_CODE_TO_EXCEPTION.get(grpc_error.code())

    error_class = error.get("class", ConnectionError) if error else ConnectionError
    error_message = error.get("message", grpc_error.details()) if error else grpc_error.details()
    details = f"Error occurred during handling the request: {error_message}"

    raise(error_class(details))


def raise_from_http(http_error):
    error = HTTP_ERROR_TYPE_TO_EXCEPTION.get(type(http_error))

    error_class = error.get("class", ConnectionError) if error else ConnectionError
    error_message = error.get("message", str(http_error)) if error else str(http_error)
    details = f"Error occurred during handling the request: {error_message}"

    raise(error_class(details))


def raise_from_http_response(error_code, error_message):
    error = HTTP_ERROR_CODE_TO_EXCEPTION.get(error_code)

    error_class = error.get("class", ModelServerError) if error else ModelServerError
    error_message = error.get("message", error_message) if error else error_message

    details = f"Error occurred during handling the request: {error_message}"

    raise(error_class(details))
