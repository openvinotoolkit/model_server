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

from tests.functional.utils.process import Process

DEFAULT_TASK_TIMEOUT = 120


def run_tasks_in_parallel_with_processes(task_command, runs, timeout=DEFAULT_TASK_TIMEOUT, print_stdout=True):
    processes = []
    for _ in range(runs):
        proc = Process()
        proc.disable_check_stderr()
        proc.policy["log-async-run"]["verbose"] = False
        proc.async_run(task_command)
        processes.append(proc)

    stdout_list = []
    for proc in processes:
        proc.wait(timeout)
        stdout, stderr = proc.get_output()
        exit_code = proc.get_exitcode()
        proc.log(exit_code, stdout, stderr, print_stdout=print_stdout)
        stdout_list.append(stdout)
    return "".join(stdout_list)


def run_command_in_parallel(task_command, runs, timeout=None, print_stdout=True):
    return run_tasks_in_parallel_with_processes(
        task_command,
        runs,
        timeout=timeout if timeout is not None else DEFAULT_TASK_TIMEOUT,
        print_stdout=print_stdout,
    )
