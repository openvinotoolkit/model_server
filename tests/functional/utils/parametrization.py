import os


def generate_port(port, position, number):
    return port[:position]+number+port[position+1:]


PORT_RANGE = os.getenv("PORT_RANGE", "1")
GRPC_PORTS = [generate_port(port=str(grpc_port), position=1, number=PORT_RANGE)
              for grpc_port in list(range(9000, 9020))]
REST_PORTS = [generate_port(port=str(rest_port), position=1, number=PORT_RANGE)
              for rest_port in list(range(5555, 5575))]


def get_ports_for_fixture():
    return {"grpc_port": GRPC_PORTS.pop(), "rest_port": REST_PORTS.pop()}


def get_tests_suffix():
    return os.environ.get("TESTS_SUFFIX", "default")
