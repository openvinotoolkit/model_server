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
    tests_ports = os.environ.get("TESTS_PORTS", "9200 9249 5700 5749")
    min_port_grpc, max_port_grpc, min_port_rest, max_port_rest = [
        int(port) for port in tests_ports.split(" ")]
    return {"grpc_port": next_free_port(min_port=min_port_grpc,
                                        max_port=max_port_grpc),
            "rest_port": next_free_port(min_port=min_port_rest,
                                        max_port=max_port_rest)}


def get_tests_suffix():
    return os.environ.get("TESTS_SUFFIX", "default")

