import pytest

from ovmsclient.tfs_compat.grpc.requests import GrpcModelStatusRequest, make_status_request
from config import MODEL_REQUEST_INVALID, MODEL_REQUEST_VALID
from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusRequest

@pytest.mark.parametrize("name, version", MODEL_REQUEST_VALID)
def test_make_status_request_valid(name, version):
    request = model_status_request = make_status_request(name, version)
    assert isinstance(request, GrpcModelStatusRequest)
    assert request.model_version == version
    assert request.model_name == name
    assert isinstance(request.raw_request, GetModelStatusRequest)

@pytest.mark.parametrize("name, version, exception, message", MODEL_REQUEST_INVALID)
def test_make_status_request_invalid(name, version, exception, message):
    with exception as e_info:
        model_status_request = make_status_request(name, version)
    assert str(e_info.value) == message
