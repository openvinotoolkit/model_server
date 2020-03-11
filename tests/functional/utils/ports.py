
def generate_port(port, position, number):
    return port[:position]+number+port[position+1:]


GRPC_PORTS = [generate_port(port=str(grpc_port), position=1, number="1")
              for grpc_port in list(range(9000, 9020))]
REST_PORTS = [generate_port(port=str(rest_port), position=1, number="1")
              for rest_port in list(range(5555, 5575))]


def get_ports_for_fixture():
    return {"grpc_port": GRPC_PORTS.pop(), "rest_port": REST_PORTS.pop()}
