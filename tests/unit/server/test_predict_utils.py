from ie_serving.server import predict_utils
import pytest
import tensorflow.contrib.util as tf_contrib_util
import numpy as np


@pytest.mark.parametrize("outputs_names, shapes, types", [
    ({'resnet': 'test'}, [(1, 1)], [np.int32]),
    ({'resnet': 'test'}, [(2, 2)], [np.float32]),
    ({'resnet': 'test'}, [(2, 2, 2)], [np.double]),
    ({'resnet': 'test', 'model': 'tensor'}, [(1, 1), (2, 2)],
     [np.double, np.int32]),
    ({'resnet': 'test', 'model': 'tensor'}, [(3, 4), (5, 6, 7)],
     [np.double, np.int32, np.float32])
])
def test_prepare_output_as_list(outputs_names, shapes, types):
    outputs = {}
    x = 0
    for key, value in outputs_names.items():
        outputs[value] = np.ones(shape=shapes[x], dtype=types[x])
        x += 1

    output = predict_utils.\
        prepare_output_as_list(inference_output=outputs,
                               model_available_outputs=outputs_names)

    x = 0
    for key, value in outputs_names.items():
        temp_output = tf_contrib_util.make_ndarray(output.outputs
                                                   [key])
        assert temp_output.shape == shapes[x]
        assert temp_output.dtype == types[x]
        x += 1


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
