#
# Copyright (c) 2020 Intel Corporation
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
import queue
import shlex
import subprocess
import threading
import time

from retry.api import retry_call

import tests.functional.config as config
from tests.functional.command_wrappers.server import start_ovms_container_command

from tests.functional.utils.grpc import port_manager_grpc
from tests.functional.utils.rest import port_manager_rest


class OvmsBinary:

    COMMON_RETRY = {"tries": 90, "delay": 2}
    GETTING_LOGS_RETRY = COMMON_RETRY
    GETTING_STATUS_RETRY = COMMON_RETRY

    def __init__(self, request, command_args, start_command, env_vars=None, cwd=None):
        self.request = request
        self.command_args = command_args
        self.grpc_port = port_manager_grpc.get_port()
        self.rest_port = port_manager_rest.get_port()
        self.command_args["port"] = self.grpc_port
        self.command_args["rest_port"] = self.rest_port
        self.command = shlex.split("./" + os.path.basename(config.ovms_binary_path) +
                                   start_ovms_container_command(start_command, self.command_args))
        self.cwd = cwd if cwd else os.path.dirname(config.ovms_binary_path)
        self.env_vars = env_vars
        self.process = None
        self.logs_flag = True

    def output_reader(self, logs_queue):
        for line in iter(self.process.stdout.readline, b''):
            if self.logs_flag:
                logs_queue.put(line)
            else:
                break

    def start(self):

        def finalizer():
            self.process.kill()
            self.logs_flag = False
            time.sleep(2)
            get_logs_thread.join()
            port_manager_grpc.release_port(self.grpc_port)
            port_manager_rest.release_port(self.rest_port)

        self.request.addfinalizer(finalizer)

        logs_queue = queue.Queue()
        get_logs_thread = threading.Thread(target=self.output_reader, args=(logs_queue,))

        env_vars_dict = {}
        if self.env_vars is not None:
            for env in self.env_vars:
                env_vars_dict[env.split("=")[0]] = env.split("=")[1]

        self.process = subprocess.Popen(self.command,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        stdin=subprocess.PIPE,
                                        universal_newlines=True,
                                        cwd=self.cwd, env=env_vars_dict)
        get_logs_thread.start()
        self.ensure_logs_contains(logs_queue)

        return self.process, {"grpc_port": self.grpc_port, "rest_port": self.rest_port}

    def stop(self):
        pass

    @staticmethod
    def ensure_logs(logs_queue):
        logs = ""
        while True:
            try:
                line = logs_queue.get(block=False)
            except queue.Empty:
                break
            logs += line
        assert config.container_log_line in logs

    def ensure_logs_contains(self, logs_queue):
        return retry_call(self.ensure_logs, exceptions=AssertionError, **OvmsBinary.GETTING_LOGS_RETRY,
                          fargs=(logs_queue,))
