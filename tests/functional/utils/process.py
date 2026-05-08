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

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from subprocess import DEVNULL, PIPE, Popen, TimeoutExpired, call, check_call
from threading import Thread
from time import sleep

import os
import psutil
from paramiko import SSHClient, WarningPolicy

from tests.functional.utils.assertions import OvmsTestException
from tests.functional.utils.logger import LoggerType, get_logger
from tests.functional.constants.os_type import get_host_os, OsType


logger = get_logger(LoggerType.SHELL_COMMAND)

TIMEOUT_CODE = 255


class AbstractProcess(ABC):
    def __init__(self):
        self.policy = {
            'log-check-output': {'exit-code': True, 'stderr': True},
            'log-run': {'verbose': True, 'trim-lines': None},
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

    def set_log_trim_lines(self, value):
        self.policy['log-run']['trim-lines'] = value

    def enable_check_stderr(self):
        self.policy['log-check-output']['stderr'] = True

    def disable_check_stderr(self):
        self.policy['log-check-output']['stderr'] = False

    def enable_check_exit_code(self):
        self.policy['log-check-output']['exit-code'] = True

    def disable_check_exit_code(self):
        self.policy['log-check-output']['exit-code'] = False

    @abstractmethod
    def async_run(self, cmd, cwd=None, daemon_mode=False, env=None, sudo=False, use_stdin=False):
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

    def run(self, cmd, cwd=None, timeout=600, env=None, sudo=False, print_stdout=True):
        self.async_run(cmd, cwd=cwd, env=env, sudo=sudo)
        exit_code, stdout, stderr = self._wait_for_output(timeout)
        self.log(exit_code, stdout, stderr, print_stdout=print_stdout)
        return exit_code, stdout, stderr

    def log(self, exit_code, stdout, stderr, print_stdout=True):
        if self.policy['log-run']['verbose']:
            trim_lines = self.policy['log-run']['trim-lines']
            stdout_log = stdout
            stderr_log = stderr
            if isinstance(trim_lines, int):
                stdout_lines = stdout.splitlines()
                stderr_lines = stderr.splitlines()
                if len(stdout_lines) > trim_lines:
                    stdout_log = "[stdout trimmed]\n" + "\n".join(stdout_lines[-trim_lines:])
                if len(stderr_lines) > trim_lines:
                    stderr_log = "[stderr trimmed]\n" + "\n".join(stderr_lines[-trim_lines:])
            if exit_code is not None and print_stdout:
                logger.info(f"Finish process: \nExit code: {exit_code}\nStdout: {stdout_log}\nStderr: {stderr_log}")
            elif exit_code is not None:
                logger.info(f"Finish process: \nExit code: {exit_code}")
            else:
                logger.info(f"Finish process: \nStdout: {stdout_log}\nStderr: {stderr_log}")

    def run_and_check_return_all(
            self,
            cmd,
            cwd=None,
            env=None,
            timeout=600,
            sudo=False,
            exception_type=OvmsTestException,
            exit_code_check=0,
            print_stdout=True,
    ):
        code, _stdout, stderr = self.run(cmd, cwd=cwd, env=env, timeout=timeout, sudo=sudo, print_stdout=print_stdout)
        self._check_output(code, _stdout, stderr, cmd, exception_type, exit_code_check)
        return code, _stdout, stderr

    def run_and_check(
            self,
            cmd,
            cwd=None,
            env=None,
            timeout=600,
            exception_type=OvmsTestException,
            exit_code_check=0,
            print_stdout=True,
    ):
        code, _stdout, stderr = self.run(cmd, cwd=cwd, env=env, timeout=timeout, print_stdout=print_stdout)
        return self._check_output(code, _stdout, stderr, cmd, exception_type, exit_code_check)

    def _check_output(self, code, _stdout, stderr, cmd, exception_type=OvmsTestException, exit_code_check=0):
        exception = None
        if self.policy['log-check-output']['exit-code'] and code != exit_code_check:
            exception = exception_type(
                f'Unexpected return code detected during executing cmd: {cmd} \n\tCode: {code}\n\tStderr: {stderr}'
            )
        if self.policy['log-check-output']['stderr'] and stderr:
            exception = exception_type(f'Detect non empty stderr during executing cmd: {cmd}\n\tStderr: {stderr}')
        if exception:
            exception.set_process_details(cmd, code, _stdout, stderr)
            raise exception
        return _stdout


class CommonProcess(AbstractProcess):
    def __init__(self):
        super().__init__()
        self._proc = None

    @staticmethod
    def _get_shell_settings():
        return False

    def async_run(
            self,
            cmd,
            cwd=None,
            daemon_mode=False,
            env=None,
            sudo=False,
            use_stdin=False,
            shell=None,
            **kwargs,
    ):
        super().async_run(cmd, cwd, daemon_mode)
        stdin_redirect = PIPE if use_stdin else DEVNULL

        cmd_mode = self._get_cmd_mode(cmd, sudo)

        self._proc = Popen(
            cmd_mode,
            stdout=PIPE,
            stderr=PIPE,
            stdin=stdin_redirect,
            cwd=cwd,
            env=env,
            universal_newlines=None,
            shell=self._get_shell_settings() if shell is None else shell,
            **kwargs,
        )

        self._err_stream = StreamReaderThread(self._proc.stderr)
        self._std_stream = StreamReaderThread(self._proc.stdout)
        self._err_stream.run()
        self._std_stream.run()

    @staticmethod
    def _get_cmd_mode(cmd, sudo):
        raise NotImplementedError

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

    def _kill_by_shell(self, pid, end_time, force=False, sudo=False):
        raise NotImplementedError

    def kill(self, force=False, timeout=30):
        logger.info(f'Killing process {self._proc.pid} (force: {force}, timeout: {timeout})!')
        end_time = datetime.now() + timedelta(seconds=timeout)
        if not self._proc:
            return None

        try:
            parent_proc = psutil.Process(self._proc.pid)
        except psutil.NoSuchProcess as e:
            return not self.is_alive()
        child_processes = parent_proc.children()
        for child_proc in child_processes:
            try:
                child_proc.terminate()
            except psutil.NoSuchProcess as e:
                pass
            except psutil.AccessDenied as e:
                self._kill_by_shell(child_proc.pid, end_time=end_time, force=force, sudo=True)

            _, alive = psutil.wait_procs([child_proc],
                                         timeout=timeout,
                                         callback=lambda proc: f'Process {proc} exit code: {proc.returncode}')
            for proc in alive:
                proc.kill()

        self._kill_by_shell(self._proc.pid, end_time, force=force)

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


class UnixProcess(CommonProcess):

    @staticmethod
    def _get_cmd_mode(cmd, sudo):
        cmd_mode = []
        if sudo:
            cmd_mode += ['sudo', '-E']

        cmd_mode += ['bash', '-c', cmd]
        return cmd_mode

    def _kill_by_shell(self, pid, end_time, force=False, sudo=False):
        if self.is_alive():
            cmd = f'kill -9 {pid}' if force else f'kill {pid}'
            if sudo:
                cmd = f"sudo {cmd}"
            check_call(cmd.split())
            cmd = f'kill -s 0 {pid}'
            while datetime.now() < end_time:
                tmp_return = call(cmd.split())
                if tmp_return == 1:
                    break
                sleep(1)


class WindowsProcess(CommonProcess):

    @staticmethod
    def _get_cmd_mode(cmd, sudo):
        if sudo:
            raise NotImplementedError
        return cmd

    @staticmethod
    def _get_shell_settings():
        return True

    def _kill_by_shell(self, pid, end_time, force=False, sudo=False):
        if self.is_alive():
            cmd = f'Taskkill /F /PID {pid}' if force else f'Taskkill /PID {pid}'
            if sudo:
                raise NotImplementedError
            while datetime.now() < end_time:
                tmp_return = call(cmd.split())
                if tmp_return == 0 or not self.is_alive():
                    break
                sleep(1)


class RemoteProcess(SSHClient, UnixProcess):
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

    def async_run(self, cmd, cwd=None, daemon_mode=False, env=None, sudo=False):
        super().async_run(cmd, cwd, daemon_mode)
        if cwd:
            cmd = f"cd {cwd} && {cmd}"
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
        logger.warning(f"Killing process (force: {force})!")
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
                logger.warning(f'Timeout detected ({self.__class__.__name__})...')
                result = TIMEOUT_CODE
        return result


Process = WindowsProcess if get_host_os() == OsType.Windows else UnixProcess


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


PID_STATE = "State"
PID_NAME = "Name"
PID_STATE_NONE = ""     # Probably insufficient access
PID_STATE_SLEEPING = "S (sleeping)"
PID_STATE_ZOMBIE = "Z (zombie)"


def get_pid_details_as_dict(pid):
    try:
        proc_status = Path(f"/proc/{pid}/status").read_text()
        proc_status_dict = {}
        for line in proc_status.splitlines():
            key, *val = line.split(":")     # if value contains multiple ':'  len(val) > 1
            proc_status_dict[key.strip()] = ":".join(val).strip()   # If len(val) > 1 again join val into string.
        return proc_status_dict
    except FileNotFoundError:
        pass    # Do not worry about it proc was killed definitly, most likely hazard during process killing
    except Exception as exc:
        logger.exception(str(exc))
    return None


def get_pid_name(pid):
    results = get_pid_details_as_dict(pid)
    name = results.get(PID_NAME, None) if results else None
    return name


def get_pid_status(pid):
    results = get_pid_details_as_dict(pid)
    state = results.get(PID_STATE, None) if results else None
    return state


class MountedShareDirectory:
    def __init__(self, share, mount_point, read_only=True, share_type="nfs"):
        self.share = share
        self.mount_point = Path(mount_point)
        self.read_only = read_only
        self.share_type = share_type
        self._proc = Process()

    def mount(self, allow_rw_permissions=False):
        if not self.mount_point.exists():
            self._proc.run_and_check(f"sudo mkdir -p {self.mount_point}")
        if list(self.mount_point.iterdir()):
            logger.warning("Skip mounting.")
            return

        if not allow_rw_permissions:
            assert self.read_only, "Dynamic mounting share with RW can lead to unfortunate events.\
                                    Please reconsider statically mounted resource prior test session."
        read_only = "-o ro" if self.read_only else ""
        cmd = f"sudo mount -t {self.share_type} {read_only} {self.share} {self.mount_point}"
        self._proc.run_and_check(cmd)

    def umount(self):
        try:
            mount_point_exists = self.mount_point.exists()
        except OSError as e:
            if "Stale file handle" in e.strerror:
                self._proc.run_and_check(f"sudo umount -l {self.mount_point}")
            else:
                raise e
        else:
            if all([mount_point_exists, list(self.mount_point.iterdir())]):
                self._proc.run_and_check(f"sudo umount -l {self.mount_point}")
            else:
                logger.warning("Skip unmounting.")

    def link(self, link_point):
        link_point_dir = os.path.dirname(link_point)
        if not os.path.exists(link_point_dir):
            logger.warning(f"Creating directory: {link_point_dir}")
            os.makedirs(link_point_dir)
        self._proc.run_and_check(f"sudo ln -s {self.mount_point} {link_point}")
