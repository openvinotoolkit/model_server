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

from pathlib import Path

from tests.functional.utils.core import SelfDeletingFileLock
from tests.functional.utils.logger import get_logger
from tests.functional.utils.process import Process

logger = get_logger(__name__)


def wget_item(dst, cmd):
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    proc = Process()
    proc.set_log_silence()
    proc.policy["log-check-output"]["stderr"] = False

    with SelfDeletingFileLock(f"{dst}.lock", self_delete=True) as _:
        proc.run_and_check(cmd)


def wget_file(url, dst):
    logger.info(f"Downloading file via wget\n{url} => {dst}")
    cmd = f"wget {url} -O {dst}"
    wget_item(dst, cmd)


def curl_file(url, dst, user, token):
    logger.info(f"Downloading file via curl\n{url} => {dst}")
    cmd = f'curl --insecure -L --user {user}:{token} "{url}" -o {dst}'
    proc = Process()
    proc.set_log_silence()
    proc.policy["log-check-output"]["stderr"] = False
    with SelfDeletingFileLock(f"{dst}.lock", self_delete=True) as _:
        proc.run_and_check(cmd)


def wget_folder(url, dst, depth=2, reject=".html,.tmp", extra_options=None):
    logger.info(f"Downloading folder via wget\n{url} => {dst}")
    options = f"-r --directory-prefix={dst} --no-parent --no-host-directories --cut-dirs={depth} --reject={reject}"
    if extra_options is not None:
        options += extra_options
    cmd = f"wget {options} {url}"
    wget_item(dst, cmd)
