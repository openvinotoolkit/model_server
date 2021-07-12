import pytest

from ovmsclient.tfs_compat.grpc.requests import _check_model_spec
from config import MODEL_REQUEST_INVALID, MODEL_REQUEST_VALID

@pytest.mark.parametrize("name, version", MODEL_REQUEST_VALID)
def test_check_model_spec_valid(name, version):
    model_status_request = _check_model_spec(name, version)

@pytest.mark.parametrize("name, version, exception, message", MODEL_REQUEST_INVALID)
def test_check_model_spec_invalid(name, version, exception, message):
    with exception as e_info:
        model_status_request = _check_model_spec(name, version)
    assert str(e_info.value) == message