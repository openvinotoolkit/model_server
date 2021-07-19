import pytest

from ovmsclient.tfs_compat.grpc.serving_client import _prepare_certs
from config import CHANNEL_CERTS_INVALID, CHANNEL_CERTS_VALID
# @pytest.mark.parametrize("client_key, client_cert, server_cert, method_call_count", CHANNEL_CERTS_VALID)
# def test_check_prepare_certs_valid(mocker, client_key, client_cert, server_cert, method_call_count):
#     mock_load_privatekey = mocker.patch('OpenSSL.crypto.load_privatekey')
#     mock_load_certificate = mocker.patch('OpenSSL.crypto.load_certificate')

#     mock_open = mocker.patch('builtins.open')

#     _prepare_certs(client_key, client_cert, server_cert)

#     assert mock_load_privatekey.call_count == method_call_count['load_privatekey']
#     assert mock_load_certificate.call_count == method_call_count['load_certificate']
#     assert mock_open.call_count == method_call_count['open']

# @pytest.mark.parametrize("client_key, client_cert, server_cert, method_call_count, expected_exception, expected_message, side_effect", CHANNEL_CERTS_INVALID)
# def test_check_prepare_certs_invalid(mocker, client_key, client_cert, server_cert, method_call_count, expected_exception, expected_message, side_effect):
#     mock_load_privatekey = mocker.patch('OpenSSL.crypto.load_privatekey', side_effect=side_effect['load_privatekey'])
#     mock_load_certificate = mocker.patch('OpenSSL.crypto.load_certificate', side_effect=side_effect['load_certificate'])

#     mock_open = mocker.patch('builtins.open')

#     with pytest.raises(expected_exception) as e_info:
#         _prepare_certs(client_key, client_cert, server_cert)
#         assert str(e_info.value) == expected_message

#     mock_load_privatekey.call_count == method_call_count['load_privatekey']
#     mock_load_certificate.call_count == method_call_count['load_certificate']
#     mock_open.call_count == method_call_count['open']

@pytest.mark.parametrize("server_cert, client_cert, client_key, method_call_count", CHANNEL_CERTS_VALID)
def test_check_prepare_certs_valid(mocker, server_cert, client_cert, client_key, method_call_count):
    mock_check_certificate_valid = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_certificate_valid')
    mock_check_key_valid = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_private_key_valid')

    _prepare_certs(server_cert, client_cert, client_key)

    assert mock_check_certificate_valid.call_count == method_call_count['check_certificate_valid']
    assert mock_check_key_valid.call_count == method_call_count['check_key_valid']

@pytest.mark.parametrize("server_cert, client_cert, client_key, method_call_count, expected_exception, expected_message, side_effect", CHANNEL_CERTS_INVALID)
def test_check_prepare_certs_valid(mocker, server_cert, client_cert, client_key, method_call_count, expected_exception, expected_message, side_effect):
    mock_check_certificate_valid = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_certificate_valid', side_effect=side_effect["check_certificate_valid"])
    mock_check_key_valid = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_private_key_valid', side_effect=side_effect["check_key_valid"])

    with pytest.raises(expected_exception) as e_info:
        _prepare_certs(server_cert, client_cert, client_key)
        assert str(e_info.value) == expected_message

    assert mock_check_certificate_valid.call_count == method_call_count['check_certificate_valid']
    assert mock_check_key_valid.call_count == method_call_count['check_key_valid']
