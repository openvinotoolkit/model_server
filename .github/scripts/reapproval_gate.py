#!/usr/bin/env python3
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reapproval Gate — checks whether significant code changes were pushed after
the most recent PR approval.

Rules:
  - If the PR has no approvals, the gate passes (nothing to protect).
  - If the PR has at least one approval and commits were pushed after the most
    recent approval, calculate code churn (additions + deletions) for those
    commits, skipping any merge-from-main commits.
  - If churn > LOC_THRESHOLD (default 10), the gate FAILS and re-approval is
    required. Review is re-requested from prior approvers.
  - Otherwise the gate PASSES.

Environment variables consumed:
  GITHUB_TOKEN       — GitHub token with pull-requests:write permission
  GITHUB_REPOSITORY  — owner/repo (set automatically by GitHub Actions)
  PR_NUMBER          — pull request number
  MAIN_BRANCH        — name of the default/main branch (default: "main")
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

LOC_THRESHOLD: int = 10


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def make_github_request(
    token: str,
    method: str,
    url: str,
    body: Optional[Dict] = None,
) -> Any:
    """Make a single GitHub API request and return the parsed JSON response."""
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "reapproval-gate/1.0")
    if body is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(body).encode("utf-8")
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code} for {method} {url}: {exc.read().decode()}", flush=True)
        raise


def github_get_paginated(token: str, base_url: str) -> List[Dict]:
    """Fetch all pages from a paginated GitHub API endpoint."""
    results: List[Dict] = []
    page = 1
    while True:
        sep = "&" if "?" in base_url else "?"
        url = f"{base_url}{sep}per_page=100&page={page}"
        data = make_github_request(token, "GET", url)
        if not data:
            break
        results.extend(data)
        if len(data) < 100:
            break
        page += 1
    return results


def get_reviews(token: str, repo: str, pr_number: int) -> List[Dict]:
    """Return all reviews for a pull request in chronological order."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews"
    return github_get_paginated(token, url)


def get_pr_commits(token: str, repo: str, pr_number: int) -> List[Dict]:
    """Return all commits for a pull request in chronological order."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/commits"
    return github_get_paginated(token, url)


def get_commit_details(token: str, repo: str, sha: str) -> Dict:
    """Return full commit details including file-level stats and parents."""
    url = f"https://api.github.com/repos/{repo}/commits/{sha}"
    return make_github_request(token, "GET", url)


def request_re_review(
    token: str,
    repo: str,
    pr_number: int,
    reviewers: List[str],
) -> None:
    """Re-request reviews from the supplied list of GitHub logins (best-effort)."""
    if not reviewers:
        return
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/requested_reviewers"
    try:
        make_github_request(token, "POST", url, {"reviewers": reviewers})
        print(f"Re-requested review from: {', '.join(reviewers)}", flush=True)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: could not re-request reviews: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------

def find_latest_approval(reviews: List[Dict]) -> Optional[Dict]:
    """
    Return the most recently submitted APPROVED review, or None if there are
    no approvals.  Reviews are assumed to be in chronological order.
    """
    latest: Optional[Dict] = None
    for review in reviews:
        if review.get("state") == "APPROVED":
            latest = review
    return latest


def is_merge_from_main(commit_data: Dict, main_branch: str = "main") -> bool:
    """
    Return True when *commit_data* represents a merge of the main branch into
    the PR branch.

    Criteria (both must hold):
      1. The commit has two or more parents (i.e. it is a merge commit).
      2. The commit message matches one of the standard patterns produced by
         ``git merge``, GitHub's "Update branch" button, or similar tooling.
    """
    parents = commit_data.get("parents", [])
    if len(parents) < 2:
        return False

    message: str = commit_data.get("commit", {}).get("message", "").strip()
    escaped = re.escape(main_branch)
    patterns = [
        rf"Merge branch '{escaped}'",
        rf"Merge remote-tracking branch 'origin/{escaped}'",
        rf"Merge remote-tracking branch 'upstream/{escaped}'",
        rf"Merge refs/heads/{escaped}",
    ]
    return any(re.search(pat, message, re.IGNORECASE) for pat in patterns)


def calculate_churn_after_approval(
    token: str,
    repo: str,
    pr_commits: List[Dict],
    approval_commit_id: str,
    main_branch: str = "main",
) -> Tuple[int, int]:
    """
    Compute code churn for commits pushed **after** *approval_commit_id*,
    ignoring any merge-from-main commits.

    Returns
    -------
    (churn, evaluated)
        churn      — total additions + deletions across non-merge commits
        evaluated  — number of non-merge commits evaluated
        Special: churn == -1 signals that the approval commit was not found.
    """
    approval_index: Optional[int] = None
    for i, commit in enumerate(pr_commits):
        if commit["sha"] == approval_commit_id:
            approval_index = i
            break

    if approval_index is None:
        return -1, 0  # approval commit not found in PR history

    commits_after = pr_commits[approval_index + 1:]
    if not commits_after:
        return 0, 0

    total_churn = 0
    evaluated = 0
    for commit_info in commits_after:
        sha: str = commit_info["sha"]
        details = get_commit_details(token, repo, sha)
        if is_merge_from_main(details, main_branch):
            print(f"  {sha[:8]}: merge from '{main_branch}' — skipped", flush=True)
            continue

        stats = details.get("stats", {})
        additions: int = stats.get("additions", 0)
        deletions: int = stats.get("deletions", 0)
        churn = additions + deletions
        print(f"  {sha[:8]}: +{additions}/-{deletions} = {churn} LOC", flush=True)
        total_churn += churn
        evaluated += 1

    return total_churn, evaluated


def run_gate(
    token: str,
    repo: str,
    pr_number: int,
    main_branch: str = "main",
) -> bool:
    """
    Execute the full reapproval-gate logic.

    Returns True (pass) or False (fail — re-approval required).
    """
    print(f"=== Reapproval Gate  PR #{pr_number}  repo: {repo} ===", flush=True)
    print(f"Main branch: {main_branch}  Threshold: {LOC_THRESHOLD} LOC", flush=True)

    reviews = get_reviews(token, repo, pr_number)
    latest_approval = find_latest_approval(reviews)

    if latest_approval is None:
        print("No approvals found — gate passes (nothing to protect).", flush=True)
        return True

    approver: str = latest_approval["user"]["login"]
    approval_commit: str = latest_approval["commit_id"]
    print(f"Latest approval: {approver!r} at commit {approval_commit}", flush=True)

    pr_commits = get_pr_commits(token, repo, pr_number)
    print(f"Total commits in PR: {len(pr_commits)}", flush=True)
    print("Evaluating commits after approval:", flush=True)

    churn, evaluated = calculate_churn_after_approval(
        token, repo, pr_commits, approval_commit, main_branch
    )

    if churn == -1:
        print(
            f"WARNING: approval commit {approval_commit} not found in PR commit history "
            "(the branch may have been force-pushed). Requiring re-approval.",
            flush=True,
        )
        approvers = list({r["user"]["login"] for r in reviews if r.get("state") == "APPROVED"})
        request_re_review(token, repo, pr_number, approvers)
        return False

    print(f"Commits evaluated after approval: {evaluated}", flush=True)
    print(f"Total churn after approval: {churn} LOC (threshold: {LOC_THRESHOLD})", flush=True)

    if churn > LOC_THRESHOLD:
        print(
            f"FAIL: {churn} LOC of changes were introduced after the most recent approval, "
            f"which exceeds the {LOC_THRESHOLD} LOC threshold. Re-approval is required.",
            flush=True,
        )
        approvers = list({r["user"]["login"] for r in reviews if r.get("state") == "APPROVED"})
        request_re_review(token, repo, pr_number, approvers)
        return False

    print(
        f"PASS: {churn} LOC changed after approval is within the {LOC_THRESHOLD} LOC threshold.",
        flush=True,
    )
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Read environment variables and run the gate; return an exit code."""
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    pr_number_str = os.environ.get("PR_NUMBER", "")
    main_branch = os.environ.get("MAIN_BRANCH", "main")

    if not token:
        print("ERROR: GITHUB_TOKEN environment variable is not set.", flush=True)
        return 1
    if not repo:
        print("ERROR: GITHUB_REPOSITORY environment variable is not set.", flush=True)
        return 1
    if not pr_number_str:
        print("ERROR: PR_NUMBER environment variable is not set.", flush=True)
        return 1
    try:
        pr_number = int(pr_number_str)
    except ValueError:
        print(f"ERROR: PR_NUMBER '{pr_number_str}' is not a valid integer.", flush=True)
        return 1

    passed = run_gate(token, repo, pr_number, main_branch)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
