"""Socket-based port checks shared by the functional test runners."""

import errno
import socket


def probe_port(port, host=""):
    """Return the bindable port (OS-selected for zero), or None if occupied."""
    with socket.socket() as sock:
        try:
            sock.bind((host, port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                return None
            raise
        return sock.getsockname()[1]