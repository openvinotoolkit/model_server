import grpc
from conftest import get_fake_model_metadata_request, PREDICT_SERVICE


def test_get_model_metadata_wrong_model_version(get_grpc_service_for_predict):
    wrong_requested_version = 999
    request = get_fake_model_metadata_request(model_name='test',
                                              metadata_field='signature_def',
                                              version=wrong_requested_version)
    grpc_server = get_grpc_service_for_predict
    rpc = grpc_server.invoke_unary_unary(
        PREDICT_SERVICE.methods_by_name['GetModelMetadata'],
        (),
        request, None)
    rpc.initial_metadata()
    response, trailing_metadata, code, details = rpc.termination()
    assert grpc.StatusCode.NOT_FOUND == code


def test_get_model_metadata_wrong_model(get_grpc_service_for_predict):
    wrong_model_name = "wrong"
    request = get_fake_model_metadata_request(model_name=wrong_model_name,
                                              metadata_field='signature_def',
                                              version=1)
    grpc_server = get_grpc_service_for_predict
    rpc = grpc_server.invoke_unary_unary(
        PREDICT_SERVICE.methods_by_name['GetModelMetadata'],
        (),
        request, None)
    rpc.initial_metadata()
    response, trailing_metadata, code, details = rpc.termination()
    assert grpc.StatusCode.NOT_FOUND == code


def test_get_model_metadata_wrong_metadata_field(get_grpc_service_for_predict):
    wrong_metadata = "wrong"
    request = get_fake_model_metadata_request(model_name='test',
                                              metadata_field=wrong_metadata,
                                              version=1)
    grpc_server = get_grpc_service_for_predict
    rpc = grpc_server.invoke_unary_unary(
        PREDICT_SERVICE.methods_by_name['GetModelMetadata'],
        (),
        request, None)
    rpc.initial_metadata()
    response, trailing_metadata, code, details = rpc.termination()
    assert grpc.StatusCode.INVALID_ARGUMENT == code


def test_get_model_metadata_correct_response(get_grpc_service_for_predict):
    request = get_fake_model_metadata_request(model_name='test',
                                              metadata_field='signature_'
                                                             'def',
                                              version=1)
    grpc_server = get_grpc_service_for_predict
    rpc = grpc_server.invoke_unary_unary(
        PREDICT_SERVICE.methods_by_name['GetModelMetadata'],
        (),
        request, None)
    rpc.initial_metadata()
    response, trailing_metadata, code, details = rpc.termination()
    assert grpc.StatusCode.OK == code
