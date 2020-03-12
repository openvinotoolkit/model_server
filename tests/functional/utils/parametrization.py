import os
import socket


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
    return {"grpc_port": next_free_port(min_port=9000, max_port=9050),
            "rest_port": next_free_port(min_port=5500, max_port=5550)}


def get_tests_suffix():
    return os.environ.get("TESTS_SUFFIX", "default")
