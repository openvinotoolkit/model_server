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

import pytest

from tests.functional.utils.reservation_manager.env_manager import EnvManager
from tests.functional.utils.reservation_manager.manager import Manager, PoolPart, Reservation
from tests.functional.utils.reservation_manager.manager_config import ManagerConfig


class TestManager:
    pool_part_ranges = [
        (2000, 3000),
        (3000, 4000),
        (4000, 5000),
        (5000, 6000),
        (6000, 7000),
        (7000, 8000),
    ]

    def test_calculate_all_pool_parts(self, mocker):
        conf_mgr = mocker.MagicMock()
        conf_mgr.pool_range_start = 20000
        conf_mgr.pool_range_stop = 60000
        conf_mgr.pool_part_size = 1000
        expected_pool_size = 40

        env_mgr = mocker.MagicMock()

        mgr = Manager(conf_mgr, env_mgr)
        mgr.calculate_all_pool_parts()

        assert len(mgr.all_pool_parts) == expected_pool_size

    def test_get_reservation_json(self, mocker):
        conf_mgr = ManagerConfig()
        env_mgr = EnvManager()
        mgr = Manager(conf_mgr, env_mgr)
        mocker.spy(env_mgr, 'get_json')

        json = mgr.get_reservation_json()

        assert "envs" in json
        assert "reservation_file" in json
        assert "shell_envs_file" in json
        assert env_mgr.get_json.call_count == 1

    def test_get_reservation_shell_envs(self, mocker):
        conf_mgr = ManagerConfig()
        env_mgr = EnvManager()
        mgr = Manager(conf_mgr, env_mgr)

        env_mgr.environment = {
            "test_key_string": "test_value",
            "test_key_with_int": 0,
        }

        expected_envs = ""
        for key, value in env_mgr.environment.items():
            expected_envs += f"export {key}='{value}'\n"
        expected_envs = expected_envs.strip()

        mocker.spy(env_mgr, 'get_shell_envs')

        shell_envs = mgr.get_reservation_shell_envs()

        assert shell_envs == expected_envs
        assert env_mgr.get_shell_envs.call_count == 1


class TestPoolPart:
    intersect_test_set = [
        ((1900, 1950), (2000, 2100), (True)),
        ((1950, 2000), (2000, 2100), (True)),
        ((1980, 2030), (2000, 2100), (False)),
        ((2030, 2040), (2000, 2100), (False)),
        ((2050, 2100), (2000, 2100), (False)),
        ((2090, 2140), (2000, 2100), (False)),
        ((2100, 2150), (2000, 2100), (True)),
        ((2150, 2200), (2000, 2100), (True)),
    ]

    def test_ranges_good(self):
        for start, stop in TestManager.pool_part_ranges:
            try:
                PoolPart(start, stop)
            except AssertionError as e:
                pytest.fail(f"Creating PoolPart should succeed with range: "
                            f"start {start}, stp: {stop}")

    def test_ranges_bad(self):
        for stop, start in TestManager.pool_part_ranges:
            with pytest.raises(AssertionError):
                PoolPart(start, stop)

    def test_is_intersect_with(self):
        for range1, range2, should_be_valid in self.intersect_test_set:
            pool_part_range1 = PoolPart(range1[0], range1[1])
            pool_part_range2 = PoolPart(range2[0], range2[1])

            if should_be_valid:
                assert not pool_part_range1.is_intersect_with(pool_part_range2)
            else:
                assert pool_part_range1.is_intersect_with(pool_part_range2)


class TestReservation:
    prefix = "reservation_manager"
    suffix = "unittests"
    reservation = Reservation

    bad_reservation_strings = [
        "singlepart",
        "double-part",
        "tri-ple-part",
        "with-four-sep-strings",
        "1000-2000-wrong-order1",
        "1000-wrong-order2-2000",
        "wrong-order3-1000-2000",
        "with-one-1000-number",
        "wrong-2000-1000-range",
    ]

    def test_validate_string_good(self):
        for start, stop in TestManager.pool_part_ranges:
            test_str = (f"{self.prefix}-" f"{start}-{stop}-" f"{self.suffix}")

            try:
                self.reservation.validate_string(test_str)
            except Exception as exc:
                pytest.fail(f"Validating string should succeed: "
                            f"string {test_str}, exception: {exc}")

    def test_validate_string_bad(self):
        for bad_string in self.bad_reservation_strings:
            with pytest.raises(AssertionError):
                self.reservation.validate_string(bad_string)

    def test_reservation_from_string_good(self):
        for start, stop in TestManager.pool_part_ranges:
            test_str = (f"{self.prefix}-" f"{start}-{stop}-" f"{self.suffix}")

            reservation = self.reservation.from_str(test_str)
            assert test_str == f"{reservation}"

    def test_reservation_from_string_bad(self):
        for stop, start in TestManager.pool_part_ranges:
            test_str = (f"{self.prefix}-" f"{start}-{stop}-" f"{self.suffix}")
            with pytest.raises(AssertionError):
                self.reservation.from_str(test_str)
