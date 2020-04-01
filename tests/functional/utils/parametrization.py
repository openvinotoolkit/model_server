import os


def get_ports_prefixes():
    ports_prefixes = os.environ.get("PORTS_PREFIX", "90 55")
    grpc_ports_prefix, rest_ports_prefix = [
        port_prefix for port_prefix in ports_prefixes.split(" ")]
    return {"grpc_port_prefix": grpc_ports_prefix,
            "rest_port_prefix": rest_ports_prefix}


def get_tests_suffix():
    return os.environ.get("TESTS_SUFFIX", "default")
