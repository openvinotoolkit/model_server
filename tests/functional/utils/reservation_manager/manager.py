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

import json
import os
import pathlib

import yaml
from filelock import Timeout

from tests.functional.utils.logger import get_logger
from tests.functional.utils.reservation_manager.env_manager import EnvManager
from tests.functional.utils.reservation_manager.exceptions import ReservationNotAvailableError, ReservationNotExistsError
from tests.functional.utils.reservation_manager.locker import Locker
from tests.functional.utils.reservation_manager.manager_config import ManagerConfig

from tests.functional.config import global_tmp_dir

logger = get_logger(__name__)


class Manager:
    """
    Class Manager is responsible for gathering information about
    possible port ranges, reserved ranges and supplying information
    for next available range.

    params::
      - res_mgr_conf - ManagerConfig
      - env_mgr      - EnvManager

    """
    def __init__(self, res_mgr_conf, env_mgr):
        """
        Get configuration from ManagerConfig and EnvManager,
        init internal variables to manage ports reservation,
        gather possible port ranges.
        """

        logger = get_logger(__name__)

        self.config = res_mgr_conf
        self.env_mgr = env_mgr

        self.all_pool_parts = []
        self.calculate_all_pool_parts()

        logger.info(f"Calculated all possible port ranges, "
                       f"len: {len(self.all_pool_parts)}")

        logger.info("Create Locker instance")
        self.reservation_lock = Locker(lock_path=os.path.join(global_tmp_dir, "res_mgr.lock"))

        self.reservations = []

        self.owned_reservation = None

    def calculate_all_pool_parts(self):
        self.all_pool_parts = []
        logger.info(
            f"Calculating all reservable port ranges with following: "
            f"pool range start: {self.config.pool_range_start}, "
            f"pool range stop: {self.config.pool_range_stop}, "
            f"pool part size: {self.config.pool_part_size}")

        for range_step in range(self.config.pool_range_start,
                                self.config.pool_range_stop,
                                self.config.pool_part_size):
            range_start = range_step
            range_stop = range_start + self.config.pool_part_size

            self.all_pool_parts.append(PoolPart(range_start, range_stop))

    def register_existing_reservations(self):
        reservation_files = [
            filename for filename in os.listdir(self.config.locks_dir)
            if os.path.isfile(os.path.join(self.config.locks_dir, filename))
            and self.config.locks_prefix in filename
        ]

        logger.info(f"Gathered reservation files: {reservation_files}; "
                       f"length: {len(reservation_files)}")

        for reservation_file in reservation_files:
            logger.info(f"Creating Reservation instance "
                           f"from string: {reservation_file}")
            try:
                reservation = Reservation.from_str(reservation_file)
                logger.info(f"Appending reservation instance "
                               f"'{reservation}' to reservations list")
                self.reservations.append(reservation)

            except AssertionError as exc:
                logger.error("While registering existing reservations")
                raise AssertionError(
                    f"Can't create reservation from string: {reservation_file}; "
                    f"exception: {exc}")

    def _create_reservation_file(self, reservation):
        """
        Create and save information about reservation file.
        Concurrently not safe.
        Exceptions:
        - pathlib.FileExistsError if file exists
        """

        reservation_lock_path = os.path.join(self.config.locks_dir, str(reservation))
        logger.info(
            f"Creating reservation lock file: {reservation_lock_path}")
        pathlib.Path(reservation_lock_path).touch(exist_ok=False)
        return reservation_lock_path

    def register_as_owned(self, reservation):
        """Registers as owned reservation in this instance."""
        self.owned_reservation = reservation
        self.env_mgr.register_reservation(reservation)

    def reserve(self, reservation_candidate):
        """Creates reservation file"""

        logger.info(f"Trying to create reservation for "
                       f"{reservation_candidate}")
        try:
            self._create_reservation_file(reservation_candidate)

        except FileExistsError as e:
            logger.critical(
                f"Creating reservation. This should have not happened, "
                f"definitely a bug: {e}")
            raise e

    def get_available_reservation(self):
        """
        Get available reservation with respect to already existing
        reservations.
        Return None if not available.
        """

        logger.info("Get existing reservations")
        self.register_existing_reservations()

        for pool_range_candidate in self.all_pool_parts:
            logger.info(f"Considering pool range: {pool_range_candidate}")
            pool_range_valid = True
            for reservation in self.reservations:
                if pool_range_candidate.is_intersect_with(reservation):
                    pool_range_valid = False
                    break

            if not pool_range_valid:
                continue

            logger.info(f"Pool range is available for reservation: "
                           f"{pool_range_candidate}")

            available_reservation = Reservation(
                self.config.locks_prefix,
                pool_range_candidate,
                self.config.reserver,
            )
            return available_reservation
        return None

    def reserve_and_return(self):
        """
        Create reservation and return.
        Concurrently safe.
        """

        logger.info("Locking reservation procedure")
        try:
            self.reservation_lock.acquire()
            logger.info("Reservation lock acquired")

            logger.info("Attempting reservation")
            reservation = self.get_available_reservation()
            if reservation:
                self.reserve(reservation)
                return reservation
            raise ReservationNotAvailableError()

        except Timeout as exc:
            raise Timeout(f"Can't lock reservation procedure: {exc}") from exc
        except Exception as exc:
            raise exc from exc
        finally:
            logger.info("Unlocking reservation procedure")
            self.reservation_lock.release()

    def get_owned_reservation(self):
        """Returns owned reservation or reserves if does not own any."""

        logger.info("Return owned reservation or create")

        if not self.owned_reservation:
            logger.info(
                "Not registered any reservation, register and return")
            reservation = self.reserve_and_return()
            self.register_as_owned(reservation)
            return reservation

        return self.owned_reservation

    def _remove_reservation_file(self, reservation):
        """Remove reservation file"""
        reservation_lock_path = os.path.join(self.config.locks_dir, str(reservation))
        logger.info(f"Removing reservation lock path: {reservation_lock_path}")
        try:
            os.remove(reservation_lock_path)
        except FileNotFoundError:
            logger.warning(f"Reservation file: {reservation_lock_path} lock already removed")

    def release_reservation(self, reservation):
        """Delete reservation file"""
        self._remove_reservation_file(reservation)

    def get_reservation_json(self):
        """Return json data for owned reservation"""

        reservation_json = self.env_mgr.get_json()
        reservation_file = f"{self.config.locks_dir}/{self.owned_reservation}"
        reservation_json["reservation_file"] = reservation_file
        reservation_json["shell_envs_file"] = self.config.reservation_file_env
        return reservation_json

    def get_reservation_shell_envs(self):
        """Return shell env variables for owned reservation"""

        reservation_shell_envs = self.env_mgr.get_shell_envs()
        return reservation_shell_envs

    def reservation_from_json(self, json_path):
        """Return reservation from json data"""

        try:
            with open(json_path, "r") as json_file:
                json_data = json.load(json_file)
                logger.info(f"json_data: {json_data}")

                reservation_lock_path = json_data["reservation_file"]
                self.config.reservation_file_env = json_data["shell_envs_file"]
                logger.info(
                    f"reservation_lock_path: {reservation_lock_path}")

                self.env_mgr.locks_dir = os.path.dirname(reservation_lock_path)
                reservation_str = os.path.basename(reservation_lock_path)

                reservation = Reservation.from_str(reservation_str)
                return reservation

        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Reservation from json: {exc}") from exc

        except json.JSONDecodeError as exc:
            raise json.JSONDecodeError(f"Loading reservation from json: {exc}") from exc

    @property
    def independent(self):
        class independent_class:
            def __init__(self, res_mgr):
                self.res_mgr = res_mgr

            def create(self, verbose=False):
                """
                Create independent reservation, save info to json and shell env
                file to filepaths pointed by
                '--reservation-file-json' and '--reservation-file-env'.
                """
                reservation = self.res_mgr.get_owned_reservation()

                reservation_json = self.res_mgr.get_reservation_json()
                json_save_path = "./reservation.json" if not (
                    self.res_mgr.config.reservation_file_json) else self.res_mgr.config.reservation_file_json

                res_shell_envs = self.res_mgr.get_reservation_shell_envs()
                shell_env_save_path = "./reservation.env" if not (
                    self.res_mgr.config.reservation_file_env) else self.res_mgr.config.reservation_file_env
                if verbose:
                    print(reservation)
                    print(reservation_json)
                    print(res_shell_envs)

                # # Open exclusively, if file exists - throw exception
                try:
                    with open(json_save_path, "x") as json_file:
                        json.dump(reservation_json,
                                  json_file,
                                  ensure_ascii=False,
                                  indent=4)
                except FileExistsError as exc:
                    self.res_mgr.release_reservation(reservation)
                    raise FileExistsError(f"Can't save reservation json: {exc}") from exc

                try:
                    with open(shell_env_save_path, "x") as shell_env_file:
                        shell_env_file.write(res_shell_envs)

                except FileExistsError as exc:
                    self.res_mgr.release_reservation(reservation)
                    raise FileExistsError(f"Can't save shell env variables: {exc}") from exc

            def remove(self):
                """
                Remove independent reservation, use file pointed by
                '--reservation-file-json' to get required information.
                """

                # Get reservation file json
                json_path = self.res_mgr.config.reservation_file_json
                try:
                    reservation = self.res_mgr.reservation_from_json(json_path)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(f"Cannot find reservation: {exc}") from exc

                reservation_file_env = self.res_mgr.config.reservation_file_env

                try:
                    self.res_mgr.release_reservation(reservation)
                except ReservationNotExistsError:
                    pass
                except Exception as exc:
                    logger.error(f"Error releasing reservation: {exc}")

                try:
                    os.remove(reservation_file_env)
                except FileNotFoundError as exc:
                    logger.warning(f"Removing reservation shell env : {exc}")

                try:
                    os.remove(json_path)
                except FileNotFoundError as exc:
                    logger.warning(f"Removing reservation json: {exc}")

            def cleanup(self):
                """Find all reservations in locks_dir, delete.
                Try to delete files pointed by
                '--reservation-file-json' and '--reservation-file-env'.
                """

                # Try with remove first because there can be reservation-file{json,env}
                try:
                    self.res_mgr.independent.remove()
                except Exception:
                    pass

                self.res_mgr.register_existing_reservations()
                existing_reservations = self.res_mgr.reservations
                for reservation in existing_reservations:
                    logger.info(f"Removing reservation: {reservation}")
                    try:
                        self.res_mgr.release_reservation(reservation)
                    except Exception as exc:
                        raise Exception(f"Can't release reservation: {exc}") from exc

        return independent_class(self)


    @staticmethod
    def manager_from_args(args):
        """
        Create and return new Manager instance
        from cli arguments or config file.

        params::
          - args - parsed argparse namespace
        """

        log = get_logger(__name__)

        log.debug(f"args: {args}")

        config_path = args.config_path
        reservation_action = None

        config = None
        try:
            with open(config_path, 'r') as file:
                config = yaml.load(file, Loader=yaml.FullLoader)["config"]

        except FileNotFoundError as exc:
            msg = f"Configuration file '{config_path}' not found"
            log.warning(msg)
            raise msg from exc

        log.debug(f"config: {config}")

        pool_range_start = 0
        pool_range_stop = 0
        pool_part_size = 0

        pool_part_ports_prefix = None
        pool_part_slices = None
        locks_dir = None

        try:
            # Port pool settings
            pool_range_start = int(config["pool_range"]["start"])
            pool_range_stop = int(config["pool_range"]["stop"])
            pool_part_size = int(config["pool_part_size"])

            # Pool parts envs settings
            pool_part_slices = config["envs"]["slices"]

            # Allow passing number of slices as int, so it may be used
            # easier when only prefix is needed.
            # This behaviour will be possibly dropped in the future.
            if isinstance(pool_part_slices, int):
                pool_part_slices = [None] * pool_part_slices

            # Don't force ports_prefix to be present
            pool_part_ports_prefix = config["envs"].get("ports_prefix", "")
            locks_dir = config["locks_dir"]

        except KeyError as exc:
            key = f"{exc}".replace("'", "")
            raise KeyError(f"Following key not found in configuration file 'config/{key}'") from exc

        except Exception as exc:
            raise Exception(f"Unknown exception while reading configuration file: {exc}") from exc

        try:  # With CLI arguments, override config file
            if args.pool_range_start:
                pool_range_start = args.pool_range_start

            if args.pool_range_stop:
                pool_range_stop = args.pool_range_stop

            if args.pool_part_size:
                pool_part_size = args.pool_part_size

            if args.locks_dir:
                locks_dir = args.locks_dir if args.locks_dir == global_tmp_dir else global_tmp_dir

            reservation_action = args.reservation_action

            reservation_file_json = args.reservation_file_json
            reservation_file_env = args.reservation_file_env

            keep_env = args.keep_env

            command = None
            reserver = "independent"

            if args.reservation_action == "command":
                command = args.reservation_command
                reserver = args.reservation_command[0].replace("-", "_")

        except Exception as exc:
            log.error(f"Getting settings from cli arguments: {exc}")
            raise Exception(f"While getting CLI arguments: {exc}") from exc

        log.info(f"All params:\n"
                  f"pool_range_start: {pool_range_start}\n"
                  f"pool_range_stop: {pool_range_stop}\n"
                  f"pool_part_size: {pool_part_size}\n"
                  f"pool_part_ports_prefix: {pool_part_ports_prefix}\n"
                  f"pool_part_slices: {pool_part_slices}\n"
                  f"locks_dir: {locks_dir}\n"
                  f"reservation_action: {reservation_action}\n"
                  f"reservation_file_json: {reservation_file_json}\n"
                  f"reservation_file_env: {reservation_file_env}\n"
                  f"command: {command}\n"
                  f"keep_env: {keep_env}\n"
                  f"reserver: {reserver}\n")

        env_mgr = EnvManager(
            pool_part_slices=pool_part_slices,
            pool_part_ports_prefix=pool_part_ports_prefix,
            keep_env=keep_env,
        )

        res_mgr_conf = ManagerConfig(
            pool_range_start=pool_range_start,
            pool_range_stop=pool_range_stop,
            pool_part_size=pool_part_size,
            locks_dir=locks_dir,
            reserver=reserver,
            port_lock_cleanup=(False if reservation_action == "create" else True),
            reservation_file_json=reservation_file_json,
            reservation_file_env=reservation_file_env,
        )

        res_mgr = Manager(res_mgr_conf, env_mgr)

        log.info(f"env_mgr: {env_mgr}")
        log.info(f"res_mgr_conf: {res_mgr_conf}")
        log.info(f"res_mgr: {res_mgr}")

        return res_mgr

class PoolPart:
    """
    PoolPart represents reservable port pool part ranges.
    Range start is inclusive, range stop is exclusive.
    """
    def __init__(self, start, stop):
        """
        Create with defined port stop and start range.
        Stop port range value is exclusive.

        Throw an exception if failed to initialize:
        1. ValueError: range boundaries are not ints
        2. AssertionError: range boundaries are invalid
        """

        self.start = int(start)
        self.stop = int(stop)
        PoolPart.validate_range(self.start, self.stop)
        self.range = range(self.start, self.stop)

    def size(self):
        return self.stop - self.start

    @staticmethod
    def validate_range(range_start, range_stop):
        assert range_stop - range_start > 0, (
            "Invalid port range, must be greater than 0: "
            f"start: {range_start}; stop: {range_stop}")

    def is_intersect_with(self, pool_part):
        """
        Check if another PoolPart or Reservation instance
        is in intersect with this instance.
        """

        if type(pool_part) is PoolPart:
            self_is_subset = (self.range.start in pool_part.range
                              or self.range[-1] in pool_part.range)
            pool_part_is_subset = (pool_part.range.start in self.range
                                   or pool_part.range[-1] in self.range)
            return self_is_subset or pool_part_is_subset

        elif type(pool_part) is Reservation:
            res_pool_part = pool_part.pool_part.range

            self_is_subset = (self.range.start in res_pool_part
                              or self.range[-1] in res_pool_part)

            reservation_is_subset = (res_pool_part.start in self.range
                                     or res_pool_part[-1] in self.range)

            return self_is_subset or reservation_is_subset

    def __str__(self):
        return f"{self.start}-{self.stop}"


class Reservation:
    """
    PoolPart reservation.

    params::
      - locks_prefix (string)   - prefix for string representation and discovery
      - pool_part    (PoolPart) - create reservation for this pool part
      - reserved_by  (string)   - suffix for string representation

    """
    def __init__(self, locks_prefix, pool_part, reserved_by):
        self.locks_prefix = locks_prefix
        self.pool_part = pool_part
        self.reserved_by = reserved_by

    def __str__(self):
        return_str = (f"{self.locks_prefix}-{self.pool_part.start}-"
                      f"{self.pool_part.stop}-{self.reserved_by}")

        try:
            Reservation.validate_string(return_str)
            return return_str
        except AssertionError as exc:
            raise AssertionError(
                f"Reservation method '__str__' returned invalid "
                f"string: {return_str}, exception: {exc}") from exc

    @staticmethod
    def validate_string(reservation):
        """Checks if string represents valid reservation"""
        elements = reservation.split("-")

        assert len(elements) == 4, (
            "Reservation string should consist of 4 elements: "
            "split with dash '-': "
            "'locks prefix', 'pool part start', "
            "'pool part stop', 'reserver identifier'")

        pool_part_start_str = elements[1]
        pool_part_stop_str = elements[2]

        try:
            pool_part_start = int(pool_part_start_str)
            pool_part_stop = int(pool_part_stop_str)
            PoolPart.validate_range(pool_part_start, pool_part_stop)
        except AssertionError as e:
            raise e

        except ValueError as exc:
            raise AssertionError(
                f"Range boundaries invalid, start: {pool_part_start_str}, "
                f"stop: {pool_part_stop_str}; exception: {exc}") from exc

    @staticmethod
    def from_str(string):
        """Create Reservation instance from string"""

        try:
            Reservation.validate_string(string)
        except AssertionError as exc:
            raise AssertionError(
                f"Validating Reservation instance from string: {string}, exception: {exc}") from exc

        values = string.split("-")
        locks_prefix = values[0]
        pool_part_start = values[1]
        pool_part_stop = values[2]
        reserved_by = values[3]

        pool_part = PoolPart(
            pool_part_start,
            pool_part_stop,
        )

        return Reservation(
            locks_prefix,
            pool_part,
            reserved_by,
        )
