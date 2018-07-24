import os
import importlib
from ie_serving import config


def test_config_default_values():
    assert config.DEVICE is "CPU"
    assert config.CPU_EXTENSION == "/opt/intel/computer_vision_sdk/" \
                                   "deployment_tools/inference_engine/lib/" \
                                   "ubuntu_16.04/intel64/" \
                                   "libcpu_extension_avx2.so"
    assert config.PLUGIN_DIR is None


def test_setting_env_variables():
    device_test_value = "device_test"
    cpu_extension_test_value = "cpu_extension_test"
    plugin_dir_test_value = "plugin_dir_test"
    os.environ['DEVICE'] = device_test_value
    os.environ['CPU_EXTENSION'] = cpu_extension_test_value
    os.environ['PLUGIN_DIR'] = plugin_dir_test_value
    importlib.reload(config)
    assert config.DEVICE == device_test_value
    assert config.PLUGIN_DIR == plugin_dir_test_value
    assert config.CPU_EXTENSION == cpu_extension_test_value
