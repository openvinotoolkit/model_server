import os
import socket
import random
from datetime import datetime

RAND_SEED = random.seed(datetime.now())


def next_free_port(min_port=1024, max_port=65535):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while min_port <= max_port:
        try:
            sock.bind(('', min_port))
            sock.close()
            return min_port
        except OSError:
            min_port += 1
    raise IOError('no free ports')


def get_ports_for_fixture():
    tests_suffix = get_tests_suffix()
    if tests_suffix == "apt-ubuntu":
        min_port_grpc = 9000
        max_port_grpc = 9049
        min_port_rest = 5500
        max_port_rest = 5549
    elif tests_suffix == "bin":
        min_port_grpc = 9050
        max_port_grpc = 9099
        min_port_rest = 5550
        max_port_rest = 5599
    elif tests_suffix == "clearlinux":
        min_port_grpc = 9100
        max_port_grpc = 9149
        min_port_rest = 5600
        max_port_rest = 5649
    elif tests_suffix == "ov-base":
        min_port_grpc = 9150
        max_port_grpc = 9199
        min_port_rest = 5650
        max_port_rest = 5699
    else:
        min_port_grpc = 9200
        max_port_grpc = 9249
        min_port_rest = 5700
        max_port_rest = 5749
    return {"grpc_port": next_free_port(min_port=min_port_grpc,
                                        max_port=max_port_grpc),
            "rest_port": next_free_port(min_port=min_port_rest,
                                        max_port=max_port_rest)}


def get_tests_suffix():
    return os.environ.get("TESTS_SUFFIX", "default")
