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

import csv
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
from dateutil import parser

from tests.functional.utils.logger import get_logger
from tests.functional.utils.process import Process
from tests.functional.config import artifacts_dir

logger = get_logger(__name__)


class ResourceMonitor(threading.Thread, ABC):
    def __init__(self):
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()
        self.join()

    def run(self):
        while not self._stop_event.is_set():
            try:
                self.check_resources()
            except StopIteration as e:
                self._stop_event.set()
                break
        self.save_data()

    @abstractmethod
    def check_resources(self):
        pass

    @abstractmethod
    def save_data(self):
        pass


def _cgroup_cache_bytes(memory_stats_stats):
    if "cache" in memory_stats_stats:
        return float(memory_stats_stats.get("cache", 0))
    return float(memory_stats_stats.get("file", 0))


class DockerResourceMonitor(ResourceMonitor):
    MEMORY_USAGE = "MEMORY_USAGE"
    PRIVATE_MEMORY = "PRIVATE_MEMORY"
    MEMORY_CACHE = "MEMORY_CACHE"
    FIELDS = ["DATE", "PIDS_COUNT", MEMORY_USAGE, PRIVATE_MEMORY, MEMORY_CACHE]  # + ["CPU_USAGE"] # Enable in further releases
    VALIDATED_FIELDS = [MEMORY_USAGE, PRIVATE_MEMORY]
    LOGGED_MEMORY_FIELDS = [MEMORY_CACHE]
    COUNTER_FIELDS = ["PIDS_COUNT"]
    LOGGED_FIELDS = LOGGED_MEMORY_FIELDS + COUNTER_FIELDS
    FIELDS_TO_STATS = {
        "DATE": lambda x: x["read"],
        "PIDS_COUNT": lambda x: int(x["pids_stats"].get("current", "0")),
        MEMORY_USAGE: lambda x: "{:.2f}M".format(float(x["memory_stats"].get("usage", "0.0")) / (2**20)),
        PRIVATE_MEMORY: lambda x: "{:.2f}M".format(
            float(x["memory_stats"].get("stats", {}).get("anon", 0)) / (2**20)
        ),
        MEMORY_CACHE: lambda x: "{:.2f}M".format(
            _cgroup_cache_bytes(x["memory_stats"].get("stats", {})) / (2**20)
        ),
        # Enable after debug & fixing
        # "CPU_USAGE": lambda x:
        #     [cpu / x['cpu_stats']['cpu_usage']['total_usage'] for cpu in x['cpu_stats']['cpu_usage']['percpu_usage']],
    }
    # Optional callback invoked after save_data with (log_path).
    on_data_saved = None

    def __init__(self, container):
        super().__init__()
        self.container = container.container
        self._docker_stats_stream = self.container.stats(stream=False, decode=False)
        self._docker_stats_data_raw = []

    def cleanup(self):
        if not self._stop_event.is_set():
            if self.is_alive():
                self.stop()
            self.save_data()

    def get_field_data(self, field, stats):
        value = DockerResourceMonitor.FIELDS_TO_STATS[field](stats)
        return value

    def save_data(self):
        self.rows = []
        # Parse raw stats
        for stats in self._docker_stats_data_raw:
            row = {}
            for field in DockerResourceMonitor.FIELDS:
                row[field] = self.get_field_data(field, stats)
            self.rows.append(row)
        log_path = Path(artifacts_dir, f"docker_stats_{self.container.name}.log")
        with log_path.open("w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=DockerResourceMonitor.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        if DockerResourceMonitor.on_data_saved:
            DockerResourceMonitor.on_data_saved(log_path)
        return log_path

    def plot_fo_file(self, x, y, field, filename):
        pass  # Uncomment when fixed regressions
        # plt.clf()  # clear trash if any
        # plt.plot(x, y)
        # plt.xlabel('Time (s)')
        # plt.title(field)
        # plt.savefig(os.path.join(artifacts_dir, filename))
        # plt.clf()  # clear after drawing

    def save_diagrams(self):
        started = parser.parse(self.rows[0]["DATE"])
        x = np.array([(parser.parse(x["DATE"]) - started).seconds for x in self.rows])
        for field in DockerResourceMonitor.FIELDS[1:]:
            y = np.array([x[field] for x in self.rows])
            filename = "diagram_{}_{}.png".format(field, self.container.name)
            self.plot_fo_file(x, y, field, filename)

    def check_resources(self):
        result = self._get_resource_data()
        self._docker_stats_data_raw.append(result)

    def _get_resource_data(self):
        """
        Example stats:
        'read' = {str} '2022-01-12T12:49:43.4286187Z'
        'preread' = {str} '0001-01-01T00:00:00Z'
        'pids_stats' = {dict: 2} {'current': 20, 'limit': 9384}
        'blkio_stats' = {dict: 8} {'io_service_bytes_recursive': ...
        'num_procs' = {int} 0
        'storage_stats' = {dict: 0} {}
        'cpu_stats' = {dict: 4} {'cpu_usage': {'total_usage': 94376000, 'percpu_usage': ...
        'precpu_stats' = {dict: 2} {'cpu_usage': {'total_usage': 0, 'usage_in_kernelmode': 0, ...
        'memory_stats' = {dict: 4} {'usage': 7974912, 'max_usage': 8007680, 'stats': {'active_anon': 3448832 ...
        'name' = {str} '/testrjasinsk_12_134942_101685'
        'id' = {str} 'b18a5c94499d60f6bb98f844883e283cb3e154b269bbc89874f112d788dc050c'
        'networks' = {dict: 1} {'eth0': {'rx_bytes': 90, 'rx_packets': 1, 'rx_errors': 0, ...
        """
        stats = self.container.stats(stream=False, decode=False)
        return stats

    def get_stats_by_field(self, field):
        result = self._get_resource_data()
        self._docker_stats_data_raw.append(result)
        return self.get_field_data(field, result)

    @classmethod
    def get_validated_metric_names(cls):
        return cls.VALIDATED_FIELDS

    @classmethod
    def get_logged_metric_names(cls):
        return cls.LOGGED_FIELDS

    @classmethod
    def get_memory_metric_names(cls):
        return cls.VALIDATED_FIELDS + cls.LOGGED_MEMORY_FIELDS

    @classmethod
    def get_counter_metric_names(cls):
        return cls.COUNTER_FIELDS


class WindowsResourceMonitor(ResourceMonitor):
    WORKING_SET_SIZE = "WORKING_SET_SIZE"
    PRIVATE_BYTES = "PRIVATE_BYTES"
    PAGE_FILE_USAGE = "PAGE_FILE_USAGE"
    PAGE_FAULTS = "PAGE_FAULTS"

    MEMORY_USAGE = WORKING_SET_SIZE

    FIELDS = ["DATE", WORKING_SET_SIZE, PRIVATE_BYTES, PAGE_FILE_USAGE, PAGE_FAULTS]
    VALIDATED_FIELDS = [WORKING_SET_SIZE, PRIVATE_BYTES]
    LOGGED_MEMORY_FIELDS = [PAGE_FILE_USAGE]
    COUNTER_FIELDS = [PAGE_FAULTS]
    LOGGED_FIELDS = LOGGED_MEMORY_FIELDS + COUNTER_FIELDS

    PS_COMMAND_TEMPLATE = (
        "powershell -NoProfile -Command \""
        "$p = Get-Process -Id {pid}; "
        "Write-Output $p.WorkingSet64; "
        "Write-Output $p.PrivateMemorySize64; "
        "Write-Output $p.PagedMemorySize64; "
        "Write-Output (Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').PageFaults\""
    )
    # Optional callback invoked after save_data with (log_path).
    on_data_saved = None

    def __init__(self, ovms_pid, proc=None):
        super().__init__()
        self.ovms_pid = ovms_pid
        self.proc = proc if proc is not None else Process()
        self._stats_data_raw = []

    def cleanup(self):
        if not self._stop_event.is_set():
            if self.is_alive():
                self.stop()
            self.save_data()

    def _get_resource_data(self):
        stats = {"DATE": datetime.now().isoformat()}
        cmd = self.PS_COMMAND_TEMPLATE.format(pid=self.ovms_pid)
        _, stdout, stderr = self.proc.run_and_check_return_all(cmd)
        lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
        if len(lines) < 4:
            raise AssertionError(
                f"Unexpected PowerShell output while collecting resource data for "
                f"pid {self.ovms_pid}: expected at least 4 non-empty lines, got "
                f"{len(lines)}. stdout={stdout!r}, stderr={stderr!r}"
            )
        stats[self.WORKING_SET_SIZE] = float(lines[0]) / (1024 * 1024)
        stats[self.PRIVATE_BYTES] = float(lines[1]) / (1024 * 1024)
        stats[self.PAGE_FILE_USAGE] = float(lines[2]) / (1024 * 1024)
        stats[self.PAGE_FAULTS] = int(lines[3])
        return stats

    def check_resources(self):
        result = self._get_resource_data()
        self._stats_data_raw.append(result)

    def save_data(self):
        self.rows = list(self._stats_data_raw)
        log_path = Path(artifacts_dir, f"windows_stats_pid_{self.ovms_pid}.log")
        with log_path.open("w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        if WindowsResourceMonitor.on_data_saved:
            WindowsResourceMonitor.on_data_saved(log_path)
        return log_path

    def get_stats_by_field(self, field):
        result = self._get_resource_data()
        self._stats_data_raw.append(result)
        value = result[field]
        if field in (self.WORKING_SET_SIZE, self.PRIVATE_BYTES, self.PAGE_FILE_USAGE):
            return f"{value:.2f}M"
        return str(value)

    @classmethod
    def get_validated_metric_names(cls):
        return cls.VALIDATED_FIELDS

    @classmethod
    def get_logged_metric_names(cls):
        return cls.LOGGED_FIELDS

    @classmethod
    def get_memory_metric_names(cls):
        return cls.VALIDATED_FIELDS + cls.LOGGED_MEMORY_FIELDS

    @classmethod
    def get_counter_metric_names(cls):
        return cls.COUNTER_FIELDS
