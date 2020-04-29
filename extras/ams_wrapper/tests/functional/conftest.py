import pytest


pytest_plugins = [
    'fixtures.ams_fixtures',
    ]


def pytest_addoption(parser):
    parser.addoption(
        "--host", action="store", default="localhost",
        help="ams wrapper host"
    )
    parser.addoption(
        "--port", action="store", default="8000",
        help="ams wrapper port"
    )


@pytest.fixture(scope="session")
def ams_host(request):
    return request.config.getoption("--host")


@pytest.fixture(scope="session")
def ams_port(request):
    return request.config.getoption("--port")