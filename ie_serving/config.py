import os

DEVICE = os.environ.get('DEVICE', "CPU")
CPU_EXTENSION = os.environ.get('CPU_EXTENSION', "/opt/intel/computer_vision_sdk/deployment_tools/"
                                                "inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so")
PLUGIN_DIR = os.environ.get('PLUGIN_DIR', None)
