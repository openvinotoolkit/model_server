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

import inspect
import os
from math import floor

from jinja2 import Template

from tests.functional.utils.logger import get_logger


class EnvManager:
    """
    Manage environment variables passed to Runner instance
    params::
      - reservation=None
      - pool_part_slices=None        - if None, default:
                                       [{
                                           "start": "TT_STARTING_PORT",
                                           "end": "TT_STOPPING_PORT",
                                           "size": "TT_PORTS_POOL_SIZE "
                                       }]
      - pool_part_ports_prefix=None
      - keep_env=False
    """
    def __init__(
            self,
            reservation=None,
            pool_part_slices=None,
            pool_part_ports_prefix=None,
            keep_env=False,
    ):
        """Create environment mapping"""
        self.keep_env = keep_env

        self.reservation = reservation

        if not pool_part_slices:
            pool_part_slices = [{
                "start": "TT_STARTING_PORT",
                "end": "TT_STOPPING_PORT",
                "size": "TT_PORTS_POOL_SIZE "
            }]
        self.pool_part_slices = pool_part_slices

        self.pool_part_ports_prefix = pool_part_ports_prefix

        self.log = get_logger(__name__)
        self.environment = {}

    def update_env_for_slice(self, slice_dict, key, value):
        """Update env but don't fail if given key does not exists"""
        try:
            self.environment.update({slice_dict[key]: str(value)})
        except (KeyError, TypeError):
            pass

    def manage_reservation_environments(self):
        """Manages registered reservation environments"""
        reservation_range_start = self.reservation.pool_part.start
        reservation_range_size = self.reservation.pool_part.size()

        reservation_slice_count = len(self.pool_part_slices)
        reservation_slice_size = floor(reservation_range_size /
                                       reservation_slice_count)

        # Save ports prefixes in this string
        ports_prefixes = ""

        for slice_index, pool_part_slice in enumerate(self.pool_part_slices):
            slice_start = (reservation_range_start +
                           slice_index * reservation_slice_size)
            slice_stop = (reservation_range_start +
                          (slice_index + 1) * reservation_slice_size)
            slice_size = slice_stop - slice_start

            ports_prefixes += f"{str(slice_start)[:-2]} "

            # Update environment but don't fail if no slice keys were provided
            self.update_env_for_slice(pool_part_slice, "start", slice_start)
            self.update_env_for_slice(pool_part_slice, "end", slice_stop)
            self.update_env_for_slice(pool_part_slice, "size", slice_size)

        # Don't force ports_prefixes to be present
        if self.pool_part_ports_prefix:
            # Trim string out of whitespace at end
            ports_prefixes = ports_prefixes[:-1]

            self.environment.update(
                {self.pool_part_ports_prefix: ports_prefixes})

    def register_reservation(self, reservation):
        """Set all attributes accordingly to given reservation"""
        self.reservation = reservation
        self.manage_reservation_environments()
        if self.keep_env:
            self.environment.update(os.environ)

    def get_json(self):
        """
        Return env variables dict with primary key set to 'envs', for example:
        ```
        "envs": {
            "POOL_START": "31000",
            "POOL_STOP": "32000",
            "TT_REST_OVMS_STARTING_PORT": "31000",
            "TT_REST_OVMS_END_PORT": "31500",
            "TT_GRPC_OVMS_STARTING_PORT": "31500",
            "TT_GRPC_OVMS_END_PORT": "32000",
            "PORTS_PREFIX": "310 315"
        }

        ```
        """
        return {'envs': self.environment}

    def get_shell_envs(self):
        """
        Return env variables in shell sourceable format,
        for example:
        ```
        export KEY1=VALUE1
        export KEY2=VALUE2
        ```
        """
        template = Template("""
                            {% for env, value in environment.items() %}
                              export {{ env }}='{{ value }}'
                            {%- endfor %}

                            """)

        exported = template.render(environment=self.environment)
        return inspect.cleandoc(exported)
