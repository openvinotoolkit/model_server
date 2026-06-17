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
from pathlib import Path

from git import InvalidGitRepositoryError, NoSuchPathError, Repo, cmd

from tests.functional.utils.assertions import GitCloneException
from tests.functional.utils.logger import get_logger

logger = get_logger(__name__)


def _get_current_git_repo_object():
    current_directory = os.getcwd()
    if not Path(current_directory, ".git").exists():
        print(f"{current_directory} isn't git repository")
        return None
    try:
        repo = Repo(current_directory, search_parent_directories=True)
    except (NoSuchPathError, InvalidGitRepositoryError) as e:
        print(f"Cannot get repo from current directory: {current_directory}")
        return None
    return repo


def get_current_branch():
    repo = _get_current_git_repo_object()
    if not repo:
        branch = ""
    else:
        try:
            branch = repo.active_branch.name
        except TypeError:
            branch = 'DETACHED_' + repo.head.object.hexsha
    return branch


def get_commit_id():
    repo = _get_current_git_repo_object()
    if repo:
        commit_id = repo.head.object.hexsha
    else:
        commit_id = ""
    return commit_id


def git_pull_repository_branch(repo_path, repo_branch):
    repo = Repo(repo_path)
    g = cmd.Git(repo_path)
    g.pull()
    repo.git.checkout(repo_branch)


def clone_git_repository(repo_url, repo_path, repo_branch, commit_sha=None):
    if not os.path.exists(repo_path):
        logger.info(f"Clone {repo_url} repository (branch {repo_branch}) to {repo_path}")
        try:
            repo = Repo.clone_from(repo_url, repo_path, branch=repo_branch)
            if commit_sha is not None:
                repo.git.checkout(commit_sha)
        except Exception as exc:
            raise GitCloneException(exc.stderr)
    return repo_path
