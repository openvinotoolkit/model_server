from ie_serving.server import predict_utils
import pytest
import tensorflow.contrib.util as tf_contrib_util
import numpy as np


@pytest.mark.parametrize("model_name, version, excepted_value", [
    ('resnet', 1, True),
    ('resnet', 0, False),
    ('resnet', 444, True),
    ('test', 1, False),
    ('test', 0, False)
])
def test_check_if_model_name_and_version_is_valid(model_name, version,
                                                  excepted_value):
    available_models_test = {'resnet': 'test', 'v50': 'test', 'v100': 'test'}
    output = predict_utils.\
        check_if_model_name_and_version_is_valid(
         model_name=model_name, version=version,
         available_models=available_models_test)
    assert output == excepted_value


@pytest.mark.parametrize("requested_model, requested_ver, expected_ver", [
    ('resnet', 1, 1),
    ('resnet', 0, 8),
    ('resnet', 444, 0),
    ('Xception', 1, 1),
    ('xception', 0, 0),
    ('xception', 6, 0)
])
def test_get_version_model(mocker, requested_model, requested_ver,
                           expected_ver):
    resnet_model_object = {'name': 'resnet', 'versions': [1, 2, 8],
                           'default_version': 8}
    inception_model_object = {'name': 'inception', 'versions': [3, 4, 5],
                              'default_version': 5}
    xception_model_object = {'name': 'Xception', 'versions': [1, 6, 8],
                             'default_version': 8}
    models = [resnet_model_object, inception_model_object,
              xception_model_object]
    available_models = {"resnet": None, 'inception': None, 'Xception': None}
    for x in models:
        model_mocker = mocker.patch('ie_serving.models.model.Model')
        model_mocker.versions = x['versions']
        model_mocker.default_version = x['default_version']
        available_models[x['name']] = model_mocker
    output = predict_utils.get_version_model(model_name=requested_model,
                                             requested_version=requested_ver,
                                             available_models=available_models)
    assert expected_ver == output


@pytest.mark.parametrize("outputs_names, shapes, types", [
    (['resnet'], [(1, 1)], [np.int32]),
    (['resnet'], [(2, 2)], [np.float32]),
    (['resnet'], [(2, 2, 2)], [np.double]),
    (['resnet', 'resnet2'], [(1, 1), (2, 2)], [np.double, np.int32]),
    (['resnet', 'resnet2'], [(3, 4), (5, 6, 7)],
     [np.double, np.int32, np.float32])
])
def test_prepare_output_as_list(outputs_names, shapes, types):
    outputs = {}
    for x in range(len(outputs_names)):
        outputs[outputs_names[x]] = np.ones(shape=shapes[x], dtype=types[x])
    output = predict_utils.\
        prepare_output_as_list(inference_output=outputs,
                               model_available_outputs=outputs_names)
    for x in range(len(outputs_names)):
        temp_output = tf_contrib_util.make_ndarray(output.outputs
                                                   [outputs_names[x]])
        assert temp_output.shape == shapes[x]
        assert temp_output.dtype == types[x]


'''
Test prepared for an unused function.
If using, please uncomment

@pytest.mark.parametrize("outputs_names, shapes, types", [
    (['resnet'], [(1, 1)], [np.int32]),
    (['resnet'], [(2, 2)], [np.float32]),
    (['resnet'], [(2, 2, 2)], [np.double]),
    (['resnet', 'resnet2'], [(1, 1), (2, 2)], [np.double, np.int32]),
    (['resnet', 'resnet2'], [(3, 4), (5, 6, 7)],
     [np.double, np.int32, np.float32])
])
def test_prepare_output_with_tf_make_tensor_proto(outputs_names, shapes,
                                                  types):
    outputs = {}
    for x in range(len(outputs_names)):
        outputs[outputs_names[x]] = np.ones(shape=shapes[x], dtype=types[x])
    output = predict_utils.\
        prepare_output_with_tf(inference_output=outputs,
                               model_available_outputs=outputs_names)
    for x in range(len(outputs_names)):
        temp_output = tf_contrib_util.make_ndarray(output.
                                                   outputs[outputs_names[x]])
        assert temp_output.shape == shapes[x]
        assert temp_output.dtype == types[x]
'''
