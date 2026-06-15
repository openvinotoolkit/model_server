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

from tests.functional.utils.logger import get_logger
from tests.functional.utils.reservation_manager.args import parse_args
from tests.functional.utils.reservation_manager.exceptions import ReservationNotAvailableError
from tests.functional.utils.reservation_manager.manager import Manager
from tests.functional.utils.reservation_manager.runner import reserve_and_run


def main():
    """Console script for manager."""

    log = get_logger(__name__)

    try:
        args = parse_args()
    except Exception as exc:
        log.error(f"While parsing arguments: {exc}")
        return 1

    if not args.reservation_action:
        log.info("Nothing to do")
        return

    log.info("Starting reservation manager")
    reservation_mgr = Manager.manager_from_args(args)

    # Manage independent reservation
    try:
        if args.reservation_action == "create":
            reservation_mgr.independent.create()

        elif args.reservation_action == "remove":
            reservation_mgr.independent.remove()

        elif args.reservation_action == "cleanup":
            reservation_mgr.independent.cleanup()

        elif args.reservation_action == "command":
            return reserve_and_run(args.reservation_command, reservation_mgr)

        else:
            raise ValueError(f"Provided reservation action "
                             f"is not allowed: {args.reservation_action}")

    except ReservationNotAvailableError as e:
        log.error(f"{e}")
        return 1

    except Exception as e:
        log.error(f"During handling reservation: {e}")
        raise e


if __name__ == "__main__":
    main()
