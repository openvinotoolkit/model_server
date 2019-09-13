import os
import pytest
import requests
import numpy as np


def input_data_downloader(numpy_url, get_test_dir):
    filename = numpy_url.split("/")[-1]
    if not os.path.exists(get_test_dir + '/' + filename):
        response = requests.get(numpy_url, stream=True)
        with open(get_test_dir + '/' + filename, 'wb') as output:
            output.write(response.content)
    imgs = np.load(get_test_dir + '/' + filename, mmap_mode='r',
                   allow_pickle=False)
    imgs = imgs.transpose((0, 3, 1, 2))  # transpose to adjust from NHWC>NCHW
    print(imgs.shape)
    return imgs


@pytest.fixture(autouse=True, scope="session")
def input_data_downloader_v1_224(get_test_dir):
    return input_data_downloader(
        'https://storage.googleapis.com/inference-eu/models_zoo/resnet_V1_50/datasets/10_v1_imgs.npy', # noqa
        get_test_dir)


@pytest.fixture(autouse=True, scope="session")
def input_data_downloader_v3_331(get_test_dir):
    return input_data_downloader(
        'https://storage.googleapis.com/inference-eu/models_zoo/pnasnet_large/datasets/10_331_v3_imgs.npy', # noqa
        get_test_dir)
