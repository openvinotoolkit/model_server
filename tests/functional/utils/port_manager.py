#
# INTEL CONFIDENTIAL
# Copyright (c) 2022 Intel Corporation
#
# The source code contained or described herein and all documents related to
# the source code ("Material") are owned by Intel Corporation or its suppliers
# or licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material contains trade secrets and proprietary
# and confidential information of Intel or its suppliers and licensors. The
# Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified,
# published, uploaded, posted, transmitted, distributed, or disclosed in any way
# without Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery
# of the Materials, either expressly, by implication, inducement, estoppel or
# otherwise. Any license under such intellectual property rights must be express
# and approved by Intel in writing.
#
import socket
import errno

from common_libs.logger import get_logger

logger = get_logger(__name__)

class SimplePortManager:
    def __init__(self, name: str):
        self.name = name

    def get_port(self):
        generated_port = self.reserve_port()
        logger.debug(f"Generated_port={generated_port}")
        return generated_port

    def reserve_port(self):
        # Idea found in SO:
        # https://unix.stackexchange.com/questions/55913/whats-the-easiest-way-to-find-an-unused-local-port
        try:
            sock = socket.socket()
            sock.bind(('', 0))
            port = sock.getsockname()[1]
            sock.close()
            logger.debug(f"Reserved port={port}")
            return port
        except socket.error as e:
            if e.errno != errno.EADDRINUSE:
                # All ports pool used?
                raise Exception("Not expected exception found in port manager {}: {}".format(self.name, e))
        return 0

    def release_port(self, port: int):
        logger.debug(f"Release port={port}")
        try:
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', port))
            sock.close()
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                raise Exception("Address has not been deleted for port manager {}".format(self.name))
            else:
                raise Exception("Not expected exception found in port manager {}: {}".format(self.name, e))
