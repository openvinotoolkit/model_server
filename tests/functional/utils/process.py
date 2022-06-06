#
# Copyright (c) 2022 Intel Corporation
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
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from paramiko import SSHClient, WarningPolicy
import psutil
from queue import Queue
from subprocess import Popen, PIPE, TimeoutExpired, check_call, call, DEVNULL
from threading import Thread
from time import sleep

logger = logging.getLogger(__name__)

TIMEOUT_CODE = 255


class AbstractProcess(ABC):
    def __init__(self):
        self.policy = {
            'log-check-output': {'exit-code': True, 'stderr': True},
            'log-run': {'verbose': True},
            'log-async-run': {'verbose': True}}
        self._std_stream = None
        self._err_stream = None

    def get_output(self):
        return self._std_stream.pop(), self._err_stream.pop()

    def set_log_verbose(self):
        self.policy['log-run']['verbose'] = True
        self.policy['log-async-run']['verbose'] = True

    def set_log_silence(self):
        self.policy['log-run']['verbose'] = False
        self.policy['log-async-run']['verbose'] = False

    @abstractmethod
    def async_run(self, cmd, cwd=None, daemon_mode=False, env=None):
        if self.policy['log-async-run']['verbose']:
            logger.info(f'Executing cmd: {cmd} (cwd: {cwd}, daemon: {daemon_mode})')

    @abstractmethod
    def kill(self, force=False, timeout=300):
        pass

    @abstractmethod
    def wait(self, timeout):
        pass

    @abstractmethod
    def get_exitcode(self):
        pass

    def _wait_for_output(self, timeout):
        self.wait(timeout)
        _stdout, stderr = self.get_output()
        return self.get_exitcode(), _stdout, stderr

    def start_daemon(self, cmd, cwd=None):
        self.async_run(cmd, cwd, daemon_mode=True)

    def run(self, cmd, cwd=None, timeout=600, env=None):
        self.async_run(cmd, cwd=cwd, env=env)
        exit_code, stdout, stderr = self._wait_for_output(timeout)
        if self.policy['log-run']['verbose']:
            logger.info(f"Finish process: \nExit code: {exit_code}\nStdout: {stdout}\nStderr: {stderr}")
        return exit_code, stdout, stderr

    def run_and_check_return_all(self, cmd, cwd=None, env=None, timeout=600):
        code, _stdout, stderr = self.run(cmd, cwd=cwd, env=env, timeout=timeout)
        self._check_output(code, _stdout, stderr, cmd)
        return code, _stdout, stderr

    def run_and_check(self, cmd, cwd=None, env=None, timeout=600, exception_type=AssertionError):
        code, _stdout, stderr = self.run(cmd, cwd=cwd, env=env, timeout=timeout)
        return self._check_output(code, _stdout, stderr, cmd, exception_type)

    def _check_output(self, code, _stdout, stderr, cmd, exception_type=AssertionError):
        if self.policy['log-check-output']['exit-code'] and code != 0:
            raise exception_type(f'Unexpected return code detected during executing cmd: {cmd} \n\tCode: {code}\n\tStderr: {stderr}')
        if self.policy['log-check-output']['stderr'] and stderr:
            raise exception_type(f'Detect non empty stderr during executing cmd: {cmd}\n\tStderr: {stderr}')
        return _stdout


class Process(AbstractProcess):
    def __init__(self):
        super().__init__()
        self._proc = None

    def async_run(self, cmd, cwd=None, daemon_mode=False, env=None):
        super().async_run(cmd, cwd, daemon_mode)
        cmd_mode = []
        cmd_mode += ['bash', '-c', cmd]

        self._proc = Popen(cmd_mode,
                           stdout=PIPE,
                           stderr=PIPE,
                           stdin=DEVNULL,
                           cwd=cwd,
                           env=env,
                           universal_newlines=(True if daemon_mode else None))

        self._err_stream = StreamReaderThread(self._proc.stderr)
        self._std_stream = StreamReaderThread(self._proc.stdout)
        self._err_stream.run()
        self._std_stream.run()

    def wait(self, timeout):
        try:
            self._proc.wait(timeout)
            self._std_stream.wait_thread_end(timeout)
            self._err_stream.wait_thread_end(timeout)
        except TimeoutExpired:
            self._std_stream.timeout_detected = True
            self._err_stream.timeout_detected = True

    def is_alive(self):
        return self._proc.poll() is None

    def timeout_detected(self):
        return self._std_stream.timeout_detected or self._err_stream.timeout_detected

    def cleanup(self):
        self._proc.terminate()

    def kill(self, force=False, timeout=300):
        logger.info(f'Killing process {self._proc.pid} (force: {force}, timeout: {timeout})!')
        end_time = datetime.now() + timedelta(seconds=timeout)
        if not self._proc:
            return None

        parent_proc = psutil.Process(self._proc.pid)
        child_processes = parent_proc.children()
        for child_proc in child_processes:
            child_proc.terminate()
            _, alive = psutil.wait_procs(child_processes,
                                         timeout=timeout,
                                         callback=lambda proc: f'Process {proc} exit code: {proc.returncode}')
            for proc in alive:
                proc.kill()

        if self.is_alive():
            cmd = f'kill -9 {self._proc.pid}' if force else f'kill {self._proc.pid}'
            check_call(cmd.split())
            cmd = 'ps --pid {}'.format(self._proc.pid)
            while datetime.now() < end_time:
                tmp_return = call(cmd.split())
                if tmp_return == 0:
                    break
                sleep(1)

        return not self.is_alive()

    def get_exitcode(self):
        result = self._proc.returncode
        if self.is_alive() and self._std_stream.timeout_detected:
            self.kill()
            if self.is_alive():
                self.kill(force=True)
                logger.warning(f'Timeout detected ({self.__class__.__name__})...')
                result = TIMEOUT_CODE
            else:
                result = self._proc.returncode
        return result


class RemoteProcess(SSHClient, Process):
    def __init__(self, hostname, username=None, password=None, port=22):
        super(SSHClient, self).__init__()
        super(RemoteProcess, self).__init__()
        self._proc_stdout = None
        self._info = {'hostname': hostname,
                      'username': username,
                      'password': password,
                      'port': port}
        self._pid = None
        self._timeout = None
        self.reconnect()

    def reconnect(self, auth_timeout=None):
        self.load_system_host_keys()
        self.set_missing_host_key_policy(WarningPolicy())
        self._info['auth_timeout'] = auth_timeout
        self._info['allow_agent'] = False
        self.connect(**self._info)

    def disconnect(self):
        self.close()

    def async_run(self, cmd, cwd=None, daemon_mode=False, env=None):
        super().async_run(cmd, cwd, daemon_mode)
        if cwd:
            cmd = 'cd {} && {}'.format(cwd, cmd)
        _, self._proc_stdout, stderr = self.exec_command(cmd, timeout=self._timeout)
        self._pid = self._proc_stdout.readline().strip()
        self._std_stream = StreamReaderThread(self._proc_stdout, False)
        self._err_stream = StreamReaderThread(stderr, False)

        self._std_stream.run()
        self._err_stream.run()

    def wait(self, timeout):
        self._std_stream.wait_thread_end(timeout)
        if timeout is None:
            self._proc_stdout.channel.recv_exit_status()

    def is_alive(self):
        return not self._proc_stdout.channel.exit_status_ready()

    def kill(self, force=False):
        if not self._pid:
            return None
        logger.warning('Killing process (force: {})!'.format(force))
        self.exec_command("kill " + ("-9 " if force else "") + str(self._pid))
        return not self.is_alive()

    def get_exitcode(self):
        result = None
        if self._proc_stdout.channel.exit_status_ready():
            result = self._proc_stdout.channel.recv_exit_status()
        if self._timeout is not None and self.is_alive():
            self.kill()
            sleep(5)
            if self.is_alive():
                self.kill(force=True)
            if self._proc_stdout.channel.exit_status_ready():
                result = self._proc_stdout.channel.recv_exit_status()
            else:
                logger.warning('Timeout detected ({})...'.format(self.__class__.__name__))
                result = TIMEOUT_CODE
        return result


class StreamReaderThread:
    def __init__(self, stream, local_thread=True):
        self._queue = Queue()
        self._thread = None
        self._stream = stream
        self.timeout_detected = False
        self._local_thread = local_thread

    def run(self):
        self._thread = Thread(target=self._enqueue_output, args=(self._stream, ))
        self._thread.daemon = True
        self._thread.start()

    def _enqueue_output(self, out):
        try:
            for line in iter(out.readline, b'' if self._local_thread else ''):
                self._queue.put(line.decode('utf8', 'ignore') if self._local_thread else line)
        except:
            self.timeout_detected = True

    def pop(self):
        result = ""
        while not self._queue.empty():
            result += self._queue.get_nowait()
        return result

    def wait_thread_end(self, timeout=None):
        self._thread.join(timeout)
        self._stream.close()
