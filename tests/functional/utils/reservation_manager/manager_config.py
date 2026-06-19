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

import os

from tests.functional.utils.logger import get_logger


# pylint: disable=too-many-instance-attributes
class ManagerConfig:
    """
    ManagerConfig

    params::
      - pool_range_start=30000
      - pool_range_stop=60000
      - pool_part_size=1000
      - locks_dir="/tmp"
      - locks_prefix="reservation_manager"
      - reserver="default"
      - port_lock_cleanup=True
      - reservation_file_json="reservation.json"
      - reservation_file_env="reservation.env"
    """

    # pylint: disable=too-many-arguments
    def __init__(
            self,
            pool_range_start=30000,
            pool_range_stop=60000,
            pool_part_size=1000,
            locks_dir="/tmp",
            locks_prefix="reservation_manager",
            reserver=None,
            port_lock_cleanup=True,
            reservation_file_json="reservation.json",
            reservation_file_env="reservation.env",
    ):

        self.pool_range_start = pool_range_start
        self.pool_range_stop = pool_range_stop
        self.pool_part_size = pool_part_size

        self.locks_dir = locks_dir
        self.locks_prefix = locks_prefix
        self.reserver = "default" if not reserver else reserver

        self.port_lock_cleanup = port_lock_cleanup
        self.reservation_file_json = reservation_file_json
        self.reservation_file_env = reservation_file_env

        self.log = get_logger(__name__)
        self.validate()

    def validate(self):
        """
        Validate ManagerConfig.
        """

        assert (self.pool_range_start is not None
                and self.pool_range_stop is not None
                and self.pool_part_size is not None
                and self.locks_dir is not None), (
                    "Following values must be provided: "
                    "pool_range_start, "
                    "pool_range_stop, "
                    "pool_part_size, "
                    "locks_dir")

        assert self.pool_range_start > 1024, (
            "Port range start must be greater than "
            "1024")
        assert self.pool_range_stop > 1024, (
            "Port range stop must be greater than "
            "1024")
        assert self.pool_range_stop - self.pool_range_start > 0, (
            "Port range must be greater than 0")

        assert self.pool_part_size <= (
            self.pool_range_stop - self.pool_range_start), (
                "Port pool size must be smaller or equal to port range")

        assert os.path.exists(self.locks_dir), (
            f"Path for locks dir must exist: {self.locks_dir}")

        assert os.access(self.locks_dir, os.W_OK), (
            f"Path for locks dir must be writeable: {self.locks_dir}")

        assert len(self.reserver.split("-")) == 1, (
            f"Reserver invalid: cannot have dashed in name: {self.reserver}")
