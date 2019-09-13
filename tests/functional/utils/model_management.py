import time
import os
import shutil
import pytest
from distutils.dir_util import copy_tree


def wait_endpoint_setup(container):
    start_time = time.time()
    tick = start_time
    running = False
    logs = ""
    while tick - start_time < 300:
        tick = time.time()
        try:
            logs = str(container.logs())
            if "Server listens on port" in logs:
                running = True
                break
        except Exception as e:
            time.sleep(1)
    print("Logs from container: ", logs)
    return running


def copy_model(model, version, destination_path):
    dir_to_cpy = destination_path + str(version)
    if not os.path.exists(dir_to_cpy):
        os.makedirs(dir_to_cpy)
        shutil.copy(model[0], dir_to_cpy + '/model.bin')
        shutil.copy(model[1], dir_to_cpy + '/model.xml')
    return dir_to_cpy


@pytest.fixture(autouse=True, scope="session")
def model_version_policy_models(get_test_dir,
                                download_two_model_versions,
                                resnet_2_out_model_downloader):
    model_ver_dir = os.path.join(get_test_dir, 'saved_models', 'model_ver')
    resnets = download_two_model_versions
    resnet_1 = os.path.dirname(resnets[0][0])
    resnet_1_dir = os.path.join(model_ver_dir, '1')
    resnet_2 = os.path.dirname(resnets[1][0])
    resnet_2_dir = os.path.join(model_ver_dir, '2')
    resnet_2_out = os.path.dirname(resnet_2_out_model_downloader[0])
    resnet_2_out_dir = os.path.join(model_ver_dir, '3')
    if not os.path.exists(model_ver_dir):
        os.makedirs(model_ver_dir)
        copy_tree(resnet_1, resnet_1_dir)
        copy_tree(resnet_2, resnet_2_dir)
        copy_tree(resnet_2_out, resnet_2_out_dir)

    return resnet_1_dir, resnet_2_dir, resnet_2_out_dir
