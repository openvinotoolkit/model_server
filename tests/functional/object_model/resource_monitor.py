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
from pathlib import Path

import numpy as np
from dateutil import parser

from tests.functional.utils.logger import get_logger
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


class DockerResourceMonitor(ResourceMonitor):
    MEMORY_USAGE = "MEMORY_USAGE"
    FIELDS = ["DATE", "PIDS_COUNT", MEMORY_USAGE]  # + ["CPU_USAGE"] # Enable in further releases
    FIELDS_TO_STATS = {
        "DATE": lambda x: x["read"],
        "PIDS_COUNT": lambda x: int(x["pids_stats"].get("current", "0")),
        MEMORY_USAGE: lambda x: "{:.2f}M".format(float(x["memory_stats"].get("usage", "0.0")) / (2**20)),
        # Enable after debug & fixing
        # "CPU_USAGE": lambda x:
        #     [cpu / x['cpu_stats']['cpu_usage']['total_usage'] for cpu in x['cpu_stats']['cpu_usage']['percpu_usage']],
    }

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
