import pytest

from ovmsclient.tfs_compat.grpc.requests import make_status_request
from config import MODEL_REQUEST_INVALID, MODEL_REQUEST_VALID

@pytest.mark.parametrize("name, version", MODEL_REQUEST_VALID)
def test_make_status_request_valid(name, version):
    model_status_request = make_status_request(name, version)

@pytest.mark.parametrize("name, version, exception, message", MODEL_REQUEST_INVALID)
def test_make_status_request_invalid(name, version, exception, message):
    with exception as e_info:
        model_status_request = make_status_request(name, version)
    assert str(e_info.value) == message
