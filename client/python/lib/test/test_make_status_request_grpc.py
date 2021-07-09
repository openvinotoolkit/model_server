import pytest

from ovmsclient.tfs_compat.grpc.requests import make_status_request

valid_input = [
    ("model_name", 1),
    ("3", 17),
    ("*.-", 0),
    ("model_name", 9223372036854775806),
]

@pytest.mark.parametrize("name, version", valid_input)
def test_make_status_request_valid(name, version):
    model_status_request = make_status_request(name, version)
    assert model_status_request.model_name == name
    assert model_status_request.model_version == version
    assert model_status_request.raw_request.model_spec.name == name
    assert model_status_request.raw_request.model_spec.version.value == version

invalid_input = [
    (12, 1, pytest.raises(TypeError), "12 has type int, but expected one of: bytes, unicode"),
    (None, -1, pytest.raises(ValueError), "model_version should be positive integer, but is negative"),
    ("model_name", "3", pytest.raises(TypeError), "'3' has type str, but expected one of: int, long"),
    ("model_name" , None, pytest.raises(TypeError), "None has type NoneType, but expected one of: int, long"),
    ("model_name", -1, pytest.raises(ValueError), "model_version should be positive integer, but is negative"),
    ("model_name", 9223372036854775809, pytest.raises(ValueError), "Value out of range: 9223372036854775809"),
]

@pytest.mark.parametrize("name, version, exception, message", invalid_input)
def test_make_status_request_invalid(name, version, exception, message):
    with exception as e_info:
        model_status_request = make_status_request(name, version)
    assert str(e_info.value) == message