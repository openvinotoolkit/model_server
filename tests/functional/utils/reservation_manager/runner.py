#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import subprocess

from tests.functional.utils.logger import get_logger
from tests.functional.utils.reservation_manager.exceptions import ReservationNotAvailableError


class Runner:
    """
    Manage running command, setup evironment variables
    accoarding to Manager.Reservation instance.
    """
    def __init__(self, command, env_mgr):
        """Prepare all necessary initialization to run command."""
        self.command = command
        self.env_mgr = env_mgr

        self.reservation = env_mgr.reservation
        self.log = get_logger(__name__)

    def run(self):
        """Run and pipe output"""

        self.log.info(f"running with command: {self.command} and "
                       f"environment: {self.env_mgr.environment}")

        popen = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=self.env_mgr.environment,
        )

        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line, end="")

        popen.communicate()
        return popen.returncode


def reserve_and_run(command, reservation_mgr):
    """Use Manager to reserve port pool and run command"""

    log = get_logger(__name__)

    try:
        # Reservation
        log.info("Trying to acquire port reservation")
        reservation = reservation_mgr.get_owned_reservation()
        log.info(f"Acquired port reservation: {reservation}")

        # Runner
        runner = Runner(command, reservation_mgr.env_mgr)

        return_code = runner.run()
        log.info(f"Command returned code: {return_code}")

        if reservation_mgr.config.port_lock_cleanup:
            log.info(f"Releasing reservation: {reservation}")
            reservation_mgr.release_reservation(reservation)

        return return_code

    except ReservationNotAvailableError as e:
        raise e

    except Exception as e:
        log.error(f"Running command with reservation failed: {e}")
        raise e
