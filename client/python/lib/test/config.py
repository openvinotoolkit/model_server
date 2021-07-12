
import pytest

MODEL_REQUEST_VALID = [
    ("model_name", 1),
    ("3", 17),
    ("*.-", 0),
    ("model_name", 9223372036854775806),
]

MODEL_REQUEST_INVALID = [
    (12, 1, pytest.raises(TypeError), "model_name should be type string, but is type int"),
    (None, -1, pytest.raises(TypeError), "model_name should be type string, but is type NoneType"),
    (None, 1, pytest.raises(TypeError), "model_name should be type string, but is type NoneType"),
    ("model_name", "3", pytest.raises(TypeError), "model_version should be type int, but is type str"),
    ("model_name" , None, pytest.raises(TypeError), "model_version should be type int, but is type NoneType"),
    ("model_name", -1, pytest.raises(ValueError), "model_version should be positive integer, but is negative"),
    ("model_name", 9223372036854775809, pytest.raises(ValueError), "model_version should have max 63 bits, but has 64"),
]
